import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class GroceryStoreMarketBasketAnalyzer:
    """
    Complete Market Basket Analysis for Grocery Store Dataset
    Designed specifically for comma-separated transaction data
    """

    def __init__(self):
        self.raw_data = None
        self.transactions = None
        self.frequent_itemsets = None
        self.rules = None
        self.item_support = {}

    def load_grocery_data(self, file_path='/home/nkubito/Data_Minig_Course/Data/USArrests.csv'):
        """
        Load grocery store transaction data
        """
        try:
            # Read the CSV file
            with open(file_path, 'r') as file:
                self.raw_data = file.read().strip()

            print("== GROCERY STORE DATASET LOADED ===")

            # Parse transactions
            lines = self.raw_data.split('\n')
            transactions = []

            for line in lines:
                # Remove quotes and split by comma
                items = [item.strip() for item in line.replace('"', '').split(',') if item.strip()]
                if len(items) >= 1:  # Keep single items too
                    transactions.append(items)

            self.transactions = transactions

            print(f"Total transactions: {len(transactions)}")

            # Basic statistics
            transaction_sizes = [len(t) for t in transactions]
            avg_size = np.mean(transaction_sizes)

            print(f"Average items per transaction: {avg_size:.1f}")
            print(f"Transaction size range: {min(transaction_sizes)} to {max(transaction_sizes)} items")

            # Get all unique items
            all_items = set()
            for transaction in transactions:
                all_items.update(transaction)

            print(f"Unique items: {len(all_items)}")
            print(f"Items: {', '.join(sorted(all_items))}")

            # Show sample transactions
            print(f"\nSample transactions:")
            for i in range(min(10, len(transactions))):
                print(f"  Transaction {i + 1}: [{', '.join(transactions[i])}]")

            return transactions

        except FileNotFoundError:
            print(f"❌ File '{file_path}' not found!")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None

    def analyze_item_frequencies(self):
        """
        Analyze individual item frequencies and popularity
        """
        if self.transactions is None:
            print("Please load data first!")
            return

        # Count item frequencies
        item_counts = Counter()
        for transaction in self.transactions:
            item_counts.update(transaction)

        print("\n" + "=" * 60)
        print("ITEM FREQUENCY ANALYSIS")
        print("=" * 60)

        total_transactions = len(self.transactions)

        print(f"\nTop 10 Most Popular Items:")
        print("-" * 40)
        for item, count in item_counts.most_common(10):
            support = count / total_transactions
            print(f"{item:15} | {count:3d} times | {support * 100:5.1f}% support")

        # Store support values
        for item, count in item_counts.items():
            self.item_support[item] = count / total_transactions

        return item_counts

    def calculate_support(self, itemset):
        """
        Calculate support for an itemset
        """
        if isinstance(itemset, str):
            itemset = [itemset]

        count = 0
        for transaction in self.transactions:
            if all(item in transaction for item in itemset):
                count += 1

        return count / len(self.transactions)

    def find_frequent_itemsets(self, min_support=0.1):
        """
        Find frequent itemsets using custom Apriori implementation
        """
        if self.transactions is None:
            print("Please load data first!")
            return None

        frequent_itemsets = {}

        # Get all unique items
        all_items = set()
        for transaction in self.transactions:
            all_items.update(transaction)

        print(f"\n" + "=" * 60)
        print("FINDING FREQUENT ITEMSETS")
        print("=" * 60)
        print(f"Minimum support threshold: {min_support * 100:.1f}%")

        # Find frequent 1-itemsets
        frequent_1_itemsets = []
        print(f"\nFrequent 1-itemsets:")
        for item in all_items:
            support = self.calculate_support([item])
            if support >= min_support:
                frequent_1_itemsets.append(frozenset([item]))
                print(f"  {item}: {support:.3f}")

        frequent_itemsets[1] = frequent_1_itemsets
        print(f"Found {len(frequent_1_itemsets)} frequent 1-itemsets")

        # Find frequent k-itemsets (k > 1)
        k = 2
        while frequent_itemsets[k - 1] and k <= 5:  # Limit to 5-itemsets for interpretability
            candidates = []

            # Generate candidates
            prev_itemsets = frequent_itemsets[k - 1]
            for i in range(len(prev_itemsets)):
                for j in range(i + 1, len(prev_itemsets)):
                    # Join two (k-1)-itemsets
                    candidate = prev_itemsets[i] | prev_itemsets[j]
                    if len(candidate) == k:
                        candidates.append(candidate)

            # Remove duplicates
            candidates = list(set(candidates))

            # Check support for candidates
            frequent_k_itemsets = []
            if candidates:
                print(f"\nFrequent {k}-itemsets:")

            for candidate in candidates:
                support = self.calculate_support(list(candidate))
                if support >= min_support:
                    frequent_k_itemsets.append(candidate)
                    items_str = ', '.join(sorted(list(candidate)))
                    print(f"  {{{items_str}}}: {support:.3f}")

            if frequent_k_itemsets:
                frequent_itemsets[k] = frequent_k_itemsets
                print(f"Found {len(frequent_k_itemsets)} frequent {k}-itemsets")
            else:
                print(f"No frequent {k}-itemsets found")
                break

            k += 1

        self.frequent_itemsets = frequent_itemsets
        return frequent_itemsets

    def generate_association_rules(self, min_confidence=0.5):
        """
        Generate association rules from frequent itemsets
        """
        if self.frequent_itemsets is None:
            self.find_frequent_itemsets()

        rules = []

        print(f"\n" + "=" * 60)
        print("GENERATING ASSOCIATION RULES")
        print("=" * 60)
        print(f"Minimum confidence threshold: {min_confidence * 100:.1f}%")

        # Generate rules from itemsets of size 2 and above
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue

            for itemset in self.frequent_itemsets[k]:
                itemset_list = list(itemset)
                itemset_support = self.calculate_support(itemset_list)

                # Generate all possible antecedent-consequent pairs
                for i in range(1, len(itemset_list)):
                    for antecedent in combinations(itemset_list, i):
                        consequent = [item for item in itemset_list if item not in antecedent]

                        antecedent_support = self.calculate_support(list(antecedent))
                        consequent_support = self.calculate_support(consequent)

                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support

                            if confidence >= min_confidence:
                                lift = confidence / consequent_support if consequent_support > 0 else 0
                                conviction = (1 - consequent_support) / (1 - confidence) if confidence < 1 else float(
                                    'inf')

                                rule = {
                                    'antecedent': list(antecedent),
                                    'consequent': consequent,
                                    'support': itemset_support,
                                    'confidence': confidence,
                                    'lift': lift,
                                    'conviction': conviction,
                                    'antecedent_support': antecedent_support,
                                    'consequent_support': consequent_support
                                }
                                rules.append(rule)

        self.rules = pd.DataFrame(rules)
        print(f"\nGenerated {len(self.rules)} association rules")
        return self.rules

    def analyze_association_rules(self, top_n=15):
        """
        Analyze and display top association rules
        """
        if self.rules is None or len(self.rules) == 0:
            print("No rules found. Try lowering the minimum confidence threshold.")
            return None

        print(f"\n" + "=" * 80)
        print("ASSOCIATION RULES ANALYSIS")
        print("=" * 80)

        # Sort by lift (indicates strength of association)
        top_rules = self.rules.nlargest(top_n, 'lift')

        print(f"\nTOP {len(top_rules)} ASSOCIATION RULES (sorted by lift):")
        print("=" * 80)

        for idx, rule in top_rules.iterrows():
            antecedent = ', '.join(rule['antecedent'])
            consequent = ', '.join(rule['consequent'])

            print(f"\nRule #{idx + 1}: {antecedent} → {consequent}")
            print(f"  Support: {rule['support']:.3f} ({rule['support'] * 100:.1f}%)")
            print(f"  Confidence: {rule['confidence']:.3f} ({rule['confidence'] * 100:.1f}%)")
            print(f"  Lift: {rule['lift']:.3f}")
            if rule['conviction'] != float('inf'):
                print(f"  Conviction: {rule['conviction']:.3f}")

            # Business interpretation
            if rule['confidence'] >= 0.8:
                strength = "Very Strong"
            elif rule['confidence'] >= 0.6:
                strength = "Strong"
            else:
                strength = "Moderate"

            print(f"  📊 Interpretation: {strength} rule - Customers buying {antecedent}")
            print(f"     have {rule['confidence'] * 100:.1f}% chance of also buying {consequent}")

            if rule['lift'] > 1.5:
                print(f"  🔥 High Impact: {rule['lift']:.1f}x more likely than random chance!")
            elif rule['lift'] > 1.2:
                print(f"  ⭐ Good Impact: {rule['lift']:.1f}x more likely than random chance")

            print("-" * 80)

        return top_rules

    def generate_business_insights(self):
        """
        Generate actionable business insights for grocery store
        """
        if self.rules is None or len(self.rules) == 0:
            print("No rules available for insights. Generate rules first.")
            return

        print(f"\n" + "=" * 80)
        print("🛒 GROCERY STORE BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)

        # High confidence rules (reliable patterns)
        high_conf_rules = self.rules[self.rules['confidence'] >= 0.7]
        print(f"\n1. 🎯 STRONG PURCHASING PATTERNS ({len(high_conf_rules)} rules with confidence ≥ 70%):")
        print("-" * 60)

        for idx, rule in high_conf_rules.head(5).iterrows():
            antecedent = ', '.join(rule['antecedent'])
            consequent = ', '.join(rule['consequent'])
            print(f"   • Customers buying {antecedent}")
            print(f"     → {rule['confidence'] * 100:.1f}% also buy {consequent}")

        # High lift rules (strong associations)
        high_lift_rules = self.rules[self.rules['lift'] >= 1.5]
        print(f"\n2. 🔥 STRONG PRODUCT ASSOCIATIONS ({len(high_lift_rules)} rules with lift ≥ 1.5):")
        print("-" * 60)

        for idx, rule in high_lift_rules.head(5).iterrows():
            antecedent = ', '.join(rule['antecedent'])
            consequent = ', '.join(rule['consequent'])
            print(f"   • {antecedent} + {consequent} bought together")
            print(f"     {rule['lift']:.1f}x more often than expected by chance")

        # Most frequent items
        if self.item_support:
            print(f"\n3. 📊 MOST POPULAR ITEMS:")
            print("-" * 60)
            top_items = sorted(self.item_support.items(), key=lambda x: x[1], reverse=True)[:5]
            for item, support in top_items:
                print(f"   • {item}: {support * 100:.1f}% of customers buy this")

        print(f"\n4. 💡 ACTIONABLE BUSINESS RECOMMENDATIONS:")
        print("-" * 60)
        print("   🏪 STORE LAYOUT:")
        print("      • Place frequently associated items near each other")
        print("      • Create 'combo zones' for high-lift product pairs")
        print("      • Position high-confidence consequent items near antecedent items")

        print("\n   🏷️  PROMOTIONAL STRATEGIES:")
        print("      • Bundle products with high lift values for combo deals")
        print("      • Offer discounts on consequent items when antecedent items are purchased")
        print("      • Create 'customers who bought X also bought Y' recommendations")

        print("\n   📦 INVENTORY MANAGEMENT:")
        print("      • Stock associated items proportionally")
        print("      • Ensure availability of consequent items when antecedent items are in demand")
        print("      • Plan joint promotions for highly associated products")

        print("\n   🎯 MARKETING CAMPAIGNS:")
        print("      • Target customers who buy antecedent items with consequent item ads")
        print("      • Create themed shopping lists based on strong associations")
        print("      • Develop cross-selling strategies using high-confidence rules")

    def create_visualizations(self):
        """
        Create comprehensive visualizations for the analysis
        """
        if self.rules is None or len(self.rules) == 0:
            print("No rules to visualize. Generate rules first.")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🛒 Grocery Store Market Basket Analysis - Complete Dashboard', fontsize=16, fontweight='bold')

        # 1. Support vs Confidence scatter plot
        scatter = axes[0, 0].scatter(self.rules['support'], self.rules['confidence'],
                                     c=self.rules['lift'], cmap='viridis', alpha=0.7, s=60)
        axes[0, 0].set_xlabel('Support')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].set_title('Support vs Confidence\n(colored by Lift)')
        cbar1 = plt.colorbar(scatter, ax=axes[0, 0])
        cbar1.set_label('Lift')

        # 2. Lift distribution
        axes[0, 1].hist(self.rules['lift'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Lift')
        axes[0, 1].set_ylabel('Number of Rules')
        axes[0, 1].set_title('Distribution of Lift Values')
        axes[0, 1].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Lift = 1 (Independence)')
        axes[0, 1].legend()

        # 3. Top rules by confidence
        if len(self.rules) >= 10:
            top_conf_rules = self.rules.nlargest(10, 'confidence')
            rule_labels = [f"Rule {i + 1}" for i in range(len(top_conf_rules))]

            bars = axes[0, 2].barh(range(len(rule_labels)), top_conf_rules['confidence'], color='orange', alpha=0.7)
            axes[0, 2].set_yticks(range(len(rule_labels)))
            axes[0, 2].set_yticklabels(rule_labels, fontsize=8)
            axes[0, 2].set_xlabel('Confidence')
            axes[0, 2].set_title('Top 10 Rules by Confidence')

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[0, 2].text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                                f'{width:.2f}', ha='left', va='center', fontsize=8)

        # 4. Item frequency
        if self.item_support:
            top_items = dict(sorted(self.item_support.items(), key=lambda x: x[1], reverse=True)[:8])
            bars = axes[1, 0].bar(range(len(top_items)), list(top_items.values()),
                                  color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 0].set_xticks(range(len(top_items)))
            axes[1, 0].set_xticklabels(list(top_items.keys()), rotation=45, ha='right')
            axes[1, 0].set_ylabel('Support')
            axes[1, 0].set_title('Most Popular Items\n(Individual Item Support)')

            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 5. Rule metrics comparison
        if len(self.rules) >= 5:
            top_5_rules = self.rules.nlargest(5, 'lift')
            x = np.arange(5)
            width = 0.25

            bars1 = axes[1, 1].bar(x - width, top_5_rules['support'], width, label='Support', alpha=0.7)
            bars2 = axes[1, 1].bar(x, top_5_rules['confidence'], width, label='Confidence', alpha=0.7)
            bars3 = axes[1, 1].bar(x + width, top_5_rules['lift'] / max(top_5_rules['lift']), width,
                                   label='Lift (normalized)', alpha=0.7)

            axes[1, 1].set_xlabel('Top 5 Rules (by Lift)')
            axes[1, 1].set_ylabel('Metric Values')
            axes[1, 1].set_title('Rule Metrics Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([f'Rule {i + 1}' for i in range(5)])
            axes[1, 1].legend()

        # 6. Support vs Lift scatter
        scatter2 = axes[1, 2].scatter(self.rules['support'], self.rules['lift'],
                                      c=self.rules['confidence'], cmap='plasma', alpha=0.7, s=60)
        axes[1, 2].set_xlabel('Support')
        axes[1, 2].set_ylabel('Lift')
        axes[1, 2].set_title('Support vs Lift\n(colored by Confidence)')
        axes[1, 2].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        cbar2 = plt.colorbar(scatter2, ax=axes[1, 2])
        cbar2.set_label('Confidence')

        plt.tight_layout()
        plt.show()

        # Additional summary statistics
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   • Total rules generated: {len(self.rules)}")
        print(f"   • Average confidence: {self.rules['confidence'].mean():.3f}")
        print(f"   • Average lift: {self.rules['lift'].mean():.3f}")
        print(f"   • Rules with lift > 1: {len(self.rules[self.rules['lift'] > 1])}")
        print(f"   • High confidence rules (>70%): {len(self.rules[self.rules['confidence'] > 0.7])}")


def main():
    """
    Main function to run complete Grocery Store Market Basket Analysis
    """
    print("=" * 80)
    print("🛒 GROCERY STORE MARKET BASKET ANALYSIS")
    print("Complete Association Rules Implementation")
    print("=" * 80)

    # Initialize analyzer
    analyzer = GroceryStoreMarketBasketAnalyzer()

    # Load grocery store data
    print("\n1. 📁 LOADING GROCERY STORE DATA...")
    transactions = analyzer.load_grocery_data()

    if transactions is None:
        print("❌ Failed to load data. Please check the file path.")
        return

    # Analyze item frequencies
    print("\n2. 📊 ANALYZING ITEM FREQUENCIES...")
    analyzer.analyze_item_frequencies()

    # Find frequent itemsets
    print("\n3. 🔍 FINDING FREQUENT ITEMSETS...")
    analyzer.find_frequent_itemsets(min_support=0.1)  # 10% minimum support

    # Generate association rules
    print("\n4. 🔗 GENERATING ASSOCIATION RULES...")
    analyzer.generate_association_rules(min_confidence=0.5)  # 50% minimum confidence

    # Analyze rules
    print("\n5. 📈 ANALYZING ASSOCIATION RULES...")
    analyzer.analyze_association_rules(top_n=10)

    # Generate business insights
    print("\n6. 💡 GENERATING BUSINESS INSIGHTS...")
    analyzer.generate_business_insights()

    # Create visualizations
    print("\n7. 📊 CREATING VISUALIZATIONS...")
    analyzer.create_visualizations()

    print("\n" + "=" * 80)
    print("✅ GROCERY STORE MARKET BASKET ANALYSIS COMPLETE!")
    print("🎯 Check the insights above for actionable business recommendations")
    print("=" * 80)


if __name__ == "__main__":
    main()
