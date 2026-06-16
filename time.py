"""Stripped-down eval.py for timing only the per-batch model inference.

Loads the npart/HT/MET flow and the particle flow, runs the same sampling
calls as eval.py inside a timed loop, and reports ms/event. No output ROOT
file, no inverse-transform bookkeeping, no per-variable unpacking.
"""

import argparse
import time
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fs_lightning import FlowLightning
from fs_npf_lightning import FlowNumPFLightning
from utils.datasetloader import FastSimDataset
from models.dpm import DPM_Solver, NoiseScheduleFlow


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-p", "--checkpoint", type=str, required=True)
parser.add_argument("-ce", "--config_evt", type=str, required=True)
parser.add_argument("-pe", "--checkpoint_evt", type=str, required=True)
parser.add_argument("-n", "--n_steps", type=int, default=25)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
parser.add_argument("-ne", "--num_events", type=int, default=2000)
parser.add_argument("-bs", "--batch_size", type=int, default=1000)
parser.add_argument("--test_path", type=str, default=None)
parser.add_argument("--warmup_batches", type=int, default=1)
parser.add_argument("--output", type=str, default="timing_results.txt")
parser.add_argument("--hardware", type=str, default="", help="Hardware tag recorded in the output file (e.g. CPU model or GPU type)")
parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker procs. 0 = in-process (recommended for small timing samples)")
args = parser.parse_args()

with open(args.config, "r") as fp:
    config = yaml.full_load(fp)
with open(args.config_evt, "r") as fp:
    npf_cfg = yaml.full_load(fp)

device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
torch.set_grad_enabled(False)

if args.cpu:
    # flex_attention's compiled create_block_mask path errors on CPU with a
    # device-mismatch in aten.index, so fall back to the dense mask path.
    for cfg in (config, npf_cfg):
        mha = cfg.get("mha_config")
        if isinstance(mha, dict) and mha.get("attn_type") == "torch-flex":
            mha["attn_type"] = "torch-meff"
            print("[setup] CPU mode: overriding mha_config.attn_type torch-flex -> torch-meff", flush=True)

print(f"[setup] device={device}, n_steps={args.n_steps}, batch_size={args.batch_size}, num_events={args.num_events}", flush=True)

print("[setup] building particle FlowLightning + loading checkpoint...", flush=True)
net = FlowLightning(config)
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
for name, param in ckpt["state_dict"].items():
    if name in net.state_dict():
        try:
            net.state_dict()[name].copy_(param)
        except Exception:
            pass
net.eval()
net.net.eval()
net.to(device)
if not args.cpu:
    net = torch.compile(net)

print("[setup] building event-level FlowNumPFLightning + loading checkpoint...", flush=True)
npf_model = FlowNumPFLightning(npf_cfg)
npf_model.load_state_dict(
    torch.load(args.checkpoint_evt, map_location="cpu", weights_only=False)["state_dict"]
)
npf_model.eval()
npf_model.to(device)
if not args.cpu:
    npf_model = torch.compile(npf_model)

test_path = config["truth_path_test"] if args.test_path is None else args.test_path
print(f"[setup] loading dataset from {test_path}...", flush=True)
dataset = FastSimDataset(
    test_path, config, reduce_ds=args.num_events, entry_start=0, mode="eval"
)
print(f"[setup] dataset loaded: {len(dataset)} events", flush=True)
loader = DataLoader(
    dataset,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)

fs_in_dim = net.net.fs_in_dim
ht_mean = dataset.var_transform_dict["ht"].shift
ht_std = dataset.var_transform_dict["ht"].scale


def model_fn(x, timestep, truth, mask, global_data):
    return (1 - timestep.view(-1, 1, 1)) * net.net(
        x, truth, mask, timestep, global_data
    ) + x


sampler = DPM_Solver(model_fn=model_fn, noise_schedule=NoiseScheduleFlow())


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_batch(batch):
    truth, truth_mask, global_data = batch
    truth = truth.to(device, non_blocking=True)
    truth_mask = truth_mask.to(device, non_blocking=True)
    global_data = global_data.to(device, non_blocking=True)

    npf_ext_shape = (truth.shape[0], 4)
    pred = npf_model.sample(
        truth, npf_ext_shape, truth_mask, global_data=global_data, method="pndm"
    ).to(device)

    n_pf_pred = (
        npf_model.var_transform_dict["npart"].inverse_transform(pred[..., 1]).round().int()
    )
    pf_ht_pred = pred[..., 0]
    pf_met_x_pred = pred[..., 2]
    pf_met_y_pred = pred[..., 3]

    bad_ht = pf_ht_pred < -ht_mean / ht_std
    pf_ht_pred = pf_ht_pred.clone()
    pf_ht_pred[bad_ht] = global_data[..., -1][bad_ht]
    n_tr = truth_mask.sum(-1).int()
    n_pf_pred = n_pf_pred.clone()
    n_pf_pred[n_pf_pred < 1] = n_tr[n_pf_pred < 1]
    n_pf_pred[n_pf_pred > 400] = n_tr[n_pf_pred > 400]

    sample_mask = torch.zeros(
        (truth.shape[0], dataset.max_particles, 2), dtype=torch.bool, device=device
    )
    sample_mask[..., 0] = truth_mask
    arange = torch.arange(dataset.max_particles, device=device).unsqueeze(0)
    sample_mask[..., 1] = arange < n_pf_pred.unsqueeze(-1)

    global_data = torch.cat(
        [
            global_data[..., :-4],
            global_data[..., -4:-2],
            pf_met_x_pred.unsqueeze(-1),
            pf_met_y_pred.unsqueeze(-1),
            global_data[..., -2:],
            net.var_transform_dict["npart"].transform(n_pf_pred.float()).unsqueeze(-1),
            pf_ht_pred.unsqueeze(-1),
        ],
        -1,
    )
    truth = torch.cat(
        [
            truth[..., :2],
            torch.sin(truth[..., 2] * 1.814).unsqueeze(-1),
            torch.cos(truth[..., 2] * 1.814).unsqueeze(-1),
            truth[..., 3:],
        ],
        -1,
    )
    sampler.sample(
        torch.randn((*truth.shape[:-1], fs_in_dim), device=device),
        truth=truth,
        mask=sample_mask,
        global_data=global_data,
        steps=args.n_steps,
        method="multistep",
        skip_type="time_uniform_flow",
        order=2,
    )


print("[setup] materializing batches from DataLoader...", flush=True)
batches = list(loader)
print(f"[setup] {len(batches)} batches ready, batch_size={args.batch_size}", flush=True)

n_warmup = min(args.warmup_batches, len(batches))
for i in range(n_warmup):
    t_w0 = time.perf_counter()
    run_batch(batches[i])
    sync()
    print(f"[warmup] batch {i+1}/{n_warmup} done in {time.perf_counter() - t_w0:.2f}s (includes JIT compile on first batch)", flush=True)

timed_batches = batches[args.warmup_batches:] if len(batches) > args.warmup_batches else batches
n_events = sum(b[0].shape[0] for b in timed_batches)
print(f"[timing] starting timed loop over {len(timed_batches)} batches ({n_events} events)", flush=True)

sync()
t0 = time.perf_counter()
for i, batch in enumerate(tqdm(timed_batches, desc="timed", unit="batch")):
    run_batch(batch)
sync()
t1 = time.perf_counter()

elapsed = t1 - t0
ms_per_event = 1000.0 * elapsed / n_events
events_per_s = n_events / elapsed

summary_lines = [
    f"device: {device.type}",
    f"hardware: {args.hardware or 'unspecified'}",
    f"test_path: {test_path}",
    f"checkpoint (part): {args.checkpoint}",
    f"checkpoint (evt):  {args.checkpoint_evt}",
    f"n_steps: {args.n_steps}, batch_size: {args.batch_size}, warmup_batches: {args.warmup_batches}",
    f"timed batches: {len(timed_batches)}, events: {n_events}, elapsed: {elapsed:.3f} s",
    f"ms/event: {ms_per_event:.3f}",
    f"events/s: {events_per_s:.2f}",
]
for line in summary_lines:
    print(line, flush=True)

with open(args.output, "w") as fout:
    for line in summary_lines:
        fout.write(line + "\n")
print(f"[output] wrote results to {args.output}", flush=True)
