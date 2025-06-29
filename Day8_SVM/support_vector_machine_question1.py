import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DecisionTreePartition:
    """
    Implementation of the decision tree partition example with 6 regions.

    The tree structure follows:
    Root: X1 < 3
    ├── Left (X1 < 3): X2 < 2
    │   ├── Left (X2 < 2): R1
    │   └── Right (X2 >= 2): X1 < 1
    │       ├── Left (X1 < 1): R2
    │       └── Right (X1 >= 1): R3
    └── Right (X1 >= 3): X2 < 4
        ├── Left (X2 < 4): X1 < 5
        │   ├── Left (X1 < 5): R4
        │   └── Right (X1 >= 5): R5
        └── Right (X2 >= 4): R6
    """

    def __init__(self):
        # Define the cutpoints
        self.t1 = 3  # First split on X1
        self.t2 = 2  # Split on X2 for left branch
        self.t3 = 4  # Split on X2 for right branch
        self.t4 = 1  # Split on X1 in upper left
        self.t5 = 5  # Split on X1 in lower right

    def classify_point(self, x1, x2):
        """
        Classify a point (x1, x2) into one of the 6 regions.

        Args:
            x1 (float): First feature value
            x2 (float): Second feature value

        Returns:
            str: Region label (R1, R2, R3, R4, R5, or R6)
        """
        if x1 < self.t1:  # X1 < 3
            if x2 < self.t2:  # X2 < 2
                return "R1"
            else:  # X2 >= 2
                if x1 < self.t4:  # X1 < 1
                    return "R2"
                else:  # X1 >= 1
                    return "R3"
        else:  # X1 >= 3
            if x2 < self.t3:  # X2 < 4
                if x1 < self.t5:  # X1 < 5
                    return "R4"
                else:  # X1 >= 5
                    return "R5"
            else:  # X2 >= 4
                return "R6"

    def get_region_boundaries(self):
        """
        Get the boundaries for each region.

        Returns:
            dict: Dictionary with region labels as keys and
                  (x1_min, x1_max, x2_min, x2_max) as values
        """
        return {
            "R1": (0, self.t1, 0, self.t2),  # X1 < 3, X2 < 2
            "R2": (0, self.t4, self.t2, 5),  # X1 < 1, X2 >= 2
            "R3": (self.t4, self.t1, self.t2, 5),  # 1 <= X1 < 3, X2 >= 2
            "R4": (self.t1, self.t5, 0, self.t3),  # 3 <= X1 < 5, X2 < 4
            "R5": (self.t5, 6, 0, self.t3),  # X1 >= 5, X2 < 4
            "R6": (self.t1, 6, self.t3, 5)  # X1 >= 3, X2 >= 4
        }

    def visualize_partition(self, figsize=(12, 5)):
        """
        Visualize the decision tree partition and the corresponding tree structure.

        Args:
            figsize (tuple): Figure size for the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left plot: Feature space partition
        self._plot_feature_space(ax1)

        # Right plot: Decision tree structure
        self._plot_decision_tree(ax2)

        plt.tight_layout()
        plt.show()

    def _plot_feature_space(self, ax):
        """Plot the 2D feature space partition."""
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 5)
        ax.set_xlabel('X₁')
        ax.set_ylabel('X₂')
        ax.set_title('2D Feature Space Partition')

        # Draw splitting lines
        ax.axvline(x=self.t1, color='red', linestyle='--', linewidth=2, label=f't₁ = {self.t1}')
        ax.axhline(y=self.t2, xmin=0, xmax=self.t1 / 6, color='red', linestyle='--', linewidth=2,
                   label=f't₂ = {self.t2}')
        ax.axhline(y=self.t3, xmin=self.t1 / 6, xmax=1, color='blue', linestyle='--', linewidth=2,
                   label=f't₃ = {self.t3}')
        ax.axvline(x=self.t4, ymin=self.t2 / 5, ymax=1, color='blue', linestyle='--', linewidth=2,
                   label=f't₄ = {self.t4}')
        ax.axvline(x=self.t5, ymin=0, ymax=self.t3 / 5, color='green', linestyle='--', linewidth=2,
                   label=f't₅ = {self.t5}')

        # Add region labels
        regions = self.get_region_boundaries()
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']

        for i, (region, (x1_min, x1_max, x2_min, x2_max)) in enumerate(regions.items()):
            # Draw rectangle for each region
            rect = patches.Rectangle((x1_min, x2_min), x1_max - x1_min, x2_max - x2_min,
                                     linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.3)
            ax.add_patch(rect)

            # Add region label
            center_x = (x1_min + x1_max) / 2
            center_y = (x2_min + x2_max) / 2
            ax.text(center_x, center_y, region, fontsize=12, fontweight='bold',
                    ha='center', va='center')

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_decision_tree(self, ax):
        """Plot the decision tree structure."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title('Decision Tree Structure')
        ax.axis('off')

        # Node positions
        nodes = {
            'root': (5, 7),
            'n1': (2.5, 5.5),  # X2 < 2
            'n2': (7.5, 5.5),  # X2 < 4
            'R1': (1, 4),
            'n3': (4, 4),  # X1 < 1
            'n4': (6, 4),  # X1 < 5
            'R6': (9, 4),
            'R2': (3, 2.5),
            'R3': (5, 2.5),
            'R4': (5.5, 2.5),
            'R5': (6.5, 2.5)
        }

        # Draw nodes
        # Internal nodes (rectangles)
        internal_nodes = {
            'root': f'X₁ < {self.t1}?',
            'n1': f'X₂ < {self.t2}?',
            'n2': f'X₂ < {self.t3}?',
            'n3': f'X₁ < {self.t4}?',
            'n4': f'X₁ < {self.t5}?'
        }

        for node, label in internal_nodes.items():
            x, y = nodes[node]
            rect = patches.Rectangle((x - 0.6, y - 0.3), 1.2, 0.6,
                                     linewidth=1, edgecolor='black', facecolor='lightblue')
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', fontsize=8)

        # Leaf nodes (circles)
        leaf_nodes = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
        for node in leaf_nodes:
            x, y = nodes[node]
            circle = patches.Circle((x, y), 0.3, linewidth=1,
                                    edgecolor='black', facecolor='lightgreen')
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw edges
        edges = [
            ('root', 'n1'), ('root', 'n2'),
            ('n1', 'R1'), ('n1', 'n3'),
            ('n2', 'n4'), ('n2', 'R6'),
            ('n3', 'R2'), ('n3', 'R3'),
            ('n4', 'R4'), ('n4', 'R5')
        ]

        for parent, child in edges:
            x1, y1 = nodes[parent]
            x2, y2 = nodes[child]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1)

        # Add edge labels
        edge_labels = [
            (('root', 'n1'), 'Yes', (-0.3, -0.3)),
            (('root', 'n2'), 'No', (0.3, -0.3)),
            (('n1', 'R1'), 'Yes', (-0.3, -0.3)),
            (('n1', 'n3'), 'No', (0.3, -0.3)),
            (('n2', 'n4'), 'Yes', (-0.3, -0.3)),
            (('n2', 'R6'), 'No', (0.3, -0.3)),
            (('n3', 'R2'), 'Yes', (-0.3, -0.3)),
            (('n3', 'R3'), 'No', (0.3, -0.3)),
            (('n4', 'R4'), 'Yes', (-0.3, -0.3)),
            (('n4', 'R5'), 'No', (0.3, -0.3))
        ]

        for (parent, child), label, (dx, dy) in edge_labels:
            x1, y1 = nodes[parent]
            x2, y2 = nodes[child]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + dx, mid_y + dy, label, fontsize=7, ha='center')


def test_classification():
    """Test the classification function with sample points."""
    dt = DecisionTreePartition()

    test_points = [
        (1.5, 1.5),  # Should be R1
        (0.5, 3),  # Should be R2
        (2, 3),  # Should be R3
        (4, 2),  # Should be R4
        (5.5, 1),  # Should be R5
        (4, 4.5)  # Should be R6
    ]

    print("Classification Results:")
    print("Point (X1, X2) -> Region")
    print("-" * 25)

    for x1, x2 in test_points:
        region = dt.classify_point(x1, x2)
        print(f"({x1:3.1f}, {x2:3.1f}) -> {region}")


def demonstrate_batch_classification():
    """Demonstrate classification of multiple random points."""
    dt = DecisionTreePartition()

    # Generate random points
    np.random.seed(42)
    n_points = 50
    x1_vals = np.random.uniform(0, 6, n_points)
    x2_vals = np.random.uniform(0, 5, n_points)

    # Classify all points
    regions = [dt.classify_point(x1, x2) for x1, x2 in zip(x1_vals, x2_vals)]

    # Count points in each region
    from collections import Counter
    region_counts = Counter(regions)

    print("\nBatch Classification Results:")
    print(f"Classified {n_points} random points:")
    for region in sorted(region_counts.keys()):
        print(f"{region}: {region_counts[region]} points")


if __name__ == "__main__":
    # Create the decision tree partition
    dt_partition = DecisionTreePartition()

    # Test individual point classification
    test_classification()

    # Demonstrate batch classification
    demonstrate_batch_classification()

    # Print region definitions
    print("\nRegion Definitions:")
    regions = dt_partition.get_region_boundaries()
    for region, (x1_min, x1_max, x2_min, x2_max) in regions.items():
        if x1_max == 6:
            x1_str = f"X₁ ≥ {x1_min}"
        else:
            x1_str = f"{x1_min} ≤ X₁ < {x1_max}" if x1_min > 0 else f"X₁ < {x1_max}"

        if x2_max == 5:
            x2_str = f"X₂ ≥ {x2_min}"
        else:
            x2_str = f"{x2_min} ≤ X₂ < {x2_max}" if x2_min > 0 else f"X₂ < {x2_max}"

        print(f"{region}: {x1_str}, {x2_str}")

    # Visualize the partition
    print("\nGenerating visualization...")
    dt_partition.visualize_partition()