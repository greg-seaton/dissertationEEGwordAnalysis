import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend (only if needed)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow


fig, ax = plt.subplots(figsize=(12, 6))

# Drawing two example branches
ys = [2.5, 1.5]  # y positions for the two example branches

for y in ys:
    # Input
    ax.add_patch(Rectangle((-4, y), 0.6, 0.3, color='skyblue'))
    ax.text(-3.7, y + 0.15, 'Input\nImage', ha='center', va='center', fontsize=8)

    # Conv2D
    ax.add_patch(Rectangle((-3, y), 0.8, 0.3, color='lightgreen'))
    ax.text(-2.6, y + 0.15, 'Conv2D', ha='center', va='center', fontsize=8)

    # MaxPool
    ax.add_patch(Rectangle((-2, y), 0.8, 0.3, color='orange'))
    ax.text(-1.6, y + 0.15, 'MaxPool', ha='center', va='center', fontsize=8)

    # Flatten
    ax.add_patch(Rectangle((-1, y), 0.8, 0.3, color='plum'))
    ax.text(-0.6, y + 0.15, 'Flatten', ha='center', va='center', fontsize=8)

# Dots in between
ax.text(-3.5, 1.0, '⋮', fontsize=18, ha='center')
ax.text(-3.8, 1.0, '×32', fontsize=9, ha='center', color='gray')

# Concatenate block
ax.add_patch(Rectangle((0.5, 1.7), 1.2, 0.6, color='lightsteelblue'))
ax.text(1.1, 2.0, 'Concatenate', ha='center', va='center', fontsize=9)

# Dense layer
ax.add_patch(Rectangle((2.2, 1.7), 1.0, 0.6, color='salmon'))
ax.text(2.7, 2.0, 'Dense\n(Softmax)', ha='center', va='center', fontsize=9)

# Output
ax.add_patch(Rectangle((3.8, 1.7), 0.8, 0.6, color='khaki'))
ax.text(4.2, 2.0, 'Output', ha='center', va='center', fontsize=9)

# Arrows from flatten to concat
for y in ys:
    ax.annotate('', xy=(0.5, y + 0.15), xytext=(-0.2, y + 0.15),
                arrowprops=dict(arrowstyle='->', color='black'))

# Arrows connecting subsequent layers
ax.annotate('', xy=(1.7, 2.0), xytext=(1.1 + 0.6, 2.0), arrowprops=dict(arrowstyle='->'))
ax.annotate('', xy=(3.2, 2.0), xytext=(2.7 + 0.5, 2.0), arrowprops=dict(arrowstyle='->'))

# Formatting
ax.set_xlim(-5, 5.5)
ax.set_ylim(0.5, 3.5)
ax.axis('off')
plt.title("CNN Structure for 32 EEG Input Images", fontsize=12)
plt.tight_layout()
plt.show()
