import json
import numpy as np
import matplotlib.pyplot as plt

# ---- Load history from file ----
with open("./result/timestamp_hist.json", "r", encoding="utf-8") as f:
    history = json.load(f)

# ---- Moving average smoothing ----
def moving_average(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode="valid")

window = 1  # change to 10 for smoothing

train_loss = moving_average(history["train_loss"], window)
valid_loss = moving_average(history["valid_loss"], window)
train_acc  = moving_average(history["train_acc"], window)
valid_acc  = moving_average(history["valid_acc"], window)

epochs = range(window, window + len(train_loss))

# ---- Marker frequency ----
mark_every = 40

# ---- Scientific plotting style ----
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 15,
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# Figure 1: LOSS
# ============================================================
fig1, ax_loss = plt.subplots(figsize=(8, 5), constrained_layout=True)

ax_loss.plot(
    epochs, train_loss,
    label="Train Loss",
    linewidth=2.0,
    markevery=mark_every
)

ax_loss.plot(
    epochs, valid_loss,
    label="Valid Loss",
    linewidth=2.0,
    linestyle="-",
    markevery=mark_every
)

ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Training Loss Curve")
ax_loss.legend(frameon=False)

ax_loss.tick_params(direction="out", length=4, width=1)

fig1.savefig("training_loss_curve.png", bbox_inches="tight")
fig1.savefig("training_loss_curve.pdf", bbox_inches="tight")


# ============================================================
# Figure 2: ACCURACY
# ============================================================
fig2, ax_acc = plt.subplots(figsize=(8, 5), constrained_layout=True)

ax_acc.plot(
    epochs, train_acc,
    label="Train Accuracy",
    linewidth=2.0,
    markevery=mark_every
)

ax_acc.plot(
    epochs, valid_acc,
    label="Valid Accuracy",
    linewidth=2.0,
    linestyle="-",
    markevery=mark_every
)

ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Training Accuracy Curve")
ax_acc.legend(frameon=False)

ax_acc.tick_params(direction="out", length=4, width=1)

fig2.savefig("training_accuracy_curve.png", bbox_inches="tight")
fig2.savefig("training_accuracy_curve.pdf", bbox_inches="tight")

plt.show()
