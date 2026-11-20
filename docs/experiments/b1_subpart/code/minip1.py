"""Mini-P1: train ArcFace on 12 valve classes, one-shot the 4 held-out ones from ISO templates.

usage: minip1.py <crops.pkl> <scratch dir> <iso dir> <mode none|cc|warp> <seed> <epochs>
"""
import json, pickle, random, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
sys.path.insert(0, sys.argv[2]); from augproto import aug
from utils.model import ArcFaceModel

pkl, _, iso_dir, mode, seed, epochs = sys.argv[1:7]; seed = int(seed); epochs = int(epochs)
HELD = {"Symbol_1": "X8074", "Symbol_3": "X8068", "Symbol_4": "2101", "Symbol_10": "X8076"}
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
data = pickle.load(open(pkl, "rb"))
train_cls = [f"Symbol_{k}" for k in range(1, 17) if f"Symbol_{k}" not in HELD]
cfg = OmegaConf.create({"fgc": {"backbone": "resnet34", "pretrained": True, "imgsz": 128,
                                "arcface": {"m": 0.5, "s": 30.0, "embed_dim": 512}}})
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = ArcFaceModel(cfg, len(train_cls)).to(dev)
norm = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
jit = transforms.ColorJitter(brightness=0.2, contrast=0.2)   # current fgc.aug.jitter
items = [(x, i) for i, c in enumerate(train_cls) for x in data["train"][c]]
rng = random.Random(seed)
from utils.augment import SubpartWarp
real = SubpartWarp(p=0.5)   # global `random`, seeded above — as in a DataLoader worker
def sample(x):
    if mode == "real": return norm(jit(real(Image.fromarray(x).convert("RGB"))))
    if mode != "none" and rng.random() < 0.5: x = aug(x, rng, mode)
    return norm(jit(Image.fromarray(x).convert("RGB")))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()
for ep in range(epochs):
    model.train(); rng.shuffle(items)
    for b in range(0, len(items), 64):
        batch = items[b:b + 64]
        xb = torch.stack([sample(x) for x, _ in batch]).to(dev); yb = torch.tensor([y for _, y in batch], device=dev)
        loss = F.cross_entropy(model(xb, yb), yb)
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def emb(imgs):
    model.eval(); out = []
    for b in range(0, len(imgs), 128):
        out.append(F.normalize(model.embed(torch.stack([norm(i.convert("RGB")) for i in imgs[b:b+128]]).to(dev))).cpu())
    return torch.cat(out)
def views(x):   # 4 quarter rotations, as GĐ4 multi-view
    im = Image.fromarray(x); return [im.rotate(r, expand=True, fillcolor=int(np.median(x))) for r in (0, 90, 180, 270)]
def oneshot(protos):
    P = emb(protos); correct = total = 0
    per = {}
    for ci, c in enumerate(HELD):
        q = data["val"][c]
        E = emb([v for x in q for v in views(x)]).view(len(q), 4, -1)
        pred = (E @ P.T).max(1).values.argmax(1)
        per[c] = float((pred == ci).float().mean()); correct += int((pred == ci).sum()); total += len(q)
    return correct / total, per
iso_acc, iso_per = oneshot([Image.open(f"{iso_dir}/{t}.png").convert("L") for t in HELD.values()])
crop_acc, crop_per = oneshot([Image.fromarray(data["train"][c][0]) for c in HELD])   # in-domain 1-shot control
# seen-class sanity: val accuracy on the 12 training classes via class centers
q = [(x, i) for i, c in enumerate(train_cls) for x in data["val"][c]]
E = emb([Image.fromarray(x) for x, _ in q]); seen = float(((E @ model.class_centers().cpu().T).argmax(1) == torch.tensor([i for _, i in q])).float().mean())
print(json.dumps({"mode": mode, "seed": seed, "epochs": epochs, "iso_acc": round(iso_acc, 4), "iso_per": {k: round(v, 3) for k, v in iso_per.items()},
                  "crop_acc": round(crop_acc, 4), "seen_val_acc": round(seen, 4), "sec": round(time.time() - t0)}))
