#!/usr/bin/env python3

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
import sys


warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


class MarketBasketAnalyzer:
    def __init__(self, data_path='/home/nkubito/Data_Minig_Course/Data/groceries.csv'):
        self.data_path = data_path
        self.transactions = []
        self.df_encoded = None
        self.frequent_itemsets = None
        self.rules = None
        self.item_frequency = None
        self.top_items = []

    def load_grocery_data(self):
        print("=" * 60)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        print(f"Loading grocery data from: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            print(f"✓ Successfully read groceries.csv from {self.data_path}")

        except FileNotFoundError:
            try:
                df_raw = pd.read_csv(self.data_path, header=None)
                lines = []
                for idx, row in df_raw.iterrows():
                    line = ','.join([str(val) for val in row.values if pd.notna(val)])
                    lines.append(line + '\n')
                print(f"✓ Successfully read groceries.csv using pandas from {self.data_path}")

            except FileNotFoundError:
                try:
                    with open('groceries.csv', 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    print("✓ Successfully read groceries.csv from current directory")

                except FileNotFoundError:
                    print("❌ groceries.csv not found in specified path or current directory.")
                    print(f"❌ Looked for: {self.data_path}")
                    print("❌ Please ensure the file exists and path is correct.")
                    return False

        for line in lines:
            items = [item.strip() for item in line.strip().split(',')
                     if item.strip() and item.strip().lower() != 'nan']
            if items:  # Only add non-empty transactions
                self.transactions.append(items)

        print(f"✓ Loaded {len(self.transactions)} transactions")

        if len(self.transactions) == 0:
            print("❌ No valid transactions found!")
            return False
        te = TransactionEncoder()
        te_ary = te.fit(self.transactions).transform(self.transactions)
        self.df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        print(f"✓ Dataset shape: {self.df_encoded.shape}")
        print(f"✓ Number of unique items: {len(self.df_encoded.columns)}")

        return True

    def analyze_data_overview(self):
        if not self.transactions:
            print("❌ No transactions to analyze")
            return False

        print("\n" + "=" * 60)
        print("DATA OVERVIEW AND STATISTICS")
        print("=" * 60)

        transaction_lengths = [len(transaction) for transaction in self.transactions]

        print(f"📊 Total transactions: {len(self.transactions):,}")
        print(f"📊 Average items per transaction: {np.mean(transaction_lengths):.2f}")
        print(f"📊 Median items per transaction: {np.median(transaction_lengths):.2f}")
        print(f"📊 Max items in a transaction: {max(transaction_lengths)}")
        print(f"📊 Min items in a transaction: {min(transaction_lengths)}")

        # Most frequent items
        all_items = [item for transaction in self.transactions for item in transaction]
        self.item_frequency = Counter(all_items)

        print(f"\n🏆 Top 20 most frequent items:")
        print("-" * 60)
        self.top_items = []
        for item, count in self.item_frequency.most_common(20):
            percentage = (count / len(self.transactions)) * 100
            print(f"{item:<35} {count:>6} ({percentage:>5.1f}%)")
            self.top_items.append((item, count, percentage))

        return True

    def visualize_top_items(self, n=15, save_plot=True):
        """Create visualization of the most frequent items"""
        if not self.top_items:
            print("❌ No item data available for plotting")
            return

        items = [item[0] for item in self.top_items[:n]]
        counts = [item[1] for item in self.top_items[:n]]

        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(items)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Items', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Top {n} Most Frequent Items in Grocery Transactions', fontsize=14, fontweight='bold')
        plt.xticks(range(len(items)), items, rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     str(count), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)

        if save_plot:
            output_dir = os.path.dirname(self.data_path)
            plt.savefig(os.path.join(output_dir, 'top_items_frequency.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {os.path.join(output_dir, 'top_items_frequency.png')}")

        plt.show()

    def find_frequent_itemsets(self, min_support=0.01):

        print("\n" + "=" * 80)
        print("TASK 1: PATTERN DISCOVERY - FREQUENT ITEMSETS AND ASSOCIATIONS")
        print("=" * 80)
        print("📋 OBJECTIVE: Identify frequent itemsets and associations between grocery items")
        print(f"🔍 Method: Apriori Algorithm with minimum support threshold: {min_support}")

        if self.df_encoded is None:
            print("❌ No encoded data available")
            return False

        # Apply Apriori algorithm
        self.frequent_itemsets = apriori(self.df_encoded,
                                         min_support=min_support,
                                         use_colnames=True)

        if len(self.frequent_itemsets) == 0:
            print("❌ No frequent itemsets found with the given minimum support.")
            print("💡 Try reducing the minimum support threshold")
            return False

        self.frequent_itemsets = self.frequent_itemsets.sort_values('support', ascending=False)

        print(f"✅ DISCOVERY RESULTS: Found {len(self.frequent_itemsets)} frequent itemsets")

        print(f"\n📊 PATTERN ANALYSIS BY ITEMSET SIZE:")
        print("-" * 80)

        total_patterns = 0
        for length in sorted(self.frequent_itemsets['itemsets'].apply(len).unique()):
            itemsets_of_length = self.frequent_itemsets[
                self.frequent_itemsets['itemsets'].apply(len) == length
                ]
            total_patterns += len(itemsets_of_length)

            print(f"\n📦 {length}-ITEMSETS (Individual items combinations): {len(itemsets_of_length)} patterns found")

            if length == 1:
                print("   → These are individual items frequently purchased")
            elif length == 2:
                print("   → These are pairs of items frequently bought together")
            elif length == 3:
                print("   → These are triplets of items frequently purchased together")
            else:
                print(f"   → These are {length}-item combinations frequently purchased together")

            print(f"   Top {min(10, len(itemsets_of_length))} patterns by support:")
            for idx, row in itemsets_of_length.head(10).iterrows():
                items = ', '.join(list(row['itemsets']))
                percentage = row['support'] * 100
                print(f"     • {items:<55} Support: {row['support']:.3f} ({percentage:.1f}% of transactions)")


        print(f"\n🔍 KEY PATTERN INSIGHTS:")
        print("-" * 50)


        single_items = self.frequent_itemsets[self.frequent_itemsets['itemsets'].apply(len) == 1]
        if len(single_items) > 0:
            top_single = single_items.iloc[0]
            item_name = list(top_single['itemsets'])[0]
            print(f"📈 Most frequent single item: '{item_name}' (in {top_single['support'] * 100:.1f}% of transactions)")


        pairs = self.frequent_itemsets[self.frequent_itemsets['itemsets'].apply(len) == 2]
        if len(pairs) > 0:
            top_pair = pairs.iloc[0]
            items = ', '.join(list(top_pair['itemsets']))
            print(f"👥 Most frequent item pair: '{items}' (in {top_pair['support'] * 100:.1f}% of transactions)")

        # Larger combinations
        larger = self.frequent_itemsets[self.frequent_itemsets['itemsets'].apply(len) >= 3]
        if len(larger) > 0:
            print(f"🛒 Complex patterns: {len(larger)} combinations of 3+ items found")
            top_large = larger.iloc[0]
            items = ', '.join(list(top_large['itemsets']))
            print(f"   Best complex pattern: '{items}' (in {top_large['support'] * 100:.1f}% of transactions)")

        print(f"\n✅ TASK 1 COMPLETED: Successfully discovered {total_patterns} frequent patterns in grocery data")

        return True

    def generate_association_rules(self, metric="confidence", min_threshold=0.1):

        print("\n" + "=" * 80)
        print("TASK 2: ASSOCIATION RULES GENERATION - 'IF A, THEN B' RELATIONSHIPS")
        print("=" * 80)
        print("📋 OBJECTIVE: Create association rules describing relationships between grocery items")
        print("📏 FORMAT: 'If customer buys A, then they will buy B' with confidence metrics")
        print(f"🎯 Method: {metric.title()} with minimum threshold: {min_threshold}")

        if self.frequent_itemsets is None:
            print("❌ Please run find_frequent_itemsets() first")
            return False
        self.rules = association_rules(self.frequent_itemsets,
                                       metric=metric,
                                       min_threshold=min_threshold)
        if len(self.rules) == 0:
            print("❌ No association rules found with the given threshold.")
            print("💡 Try reducing the minimum threshold")
            return False

        self.rules = self.rules.sort_values(['confidence', 'lift'], ascending=False)

        print(f"✅ RULE GENERATION RESULTS: Created {len(self.rules)} association rules")

        print(f"\n📊 RULE METRICS EXPLANATION:")
        print("-" * 60)
        print("• SUPPORT: How frequently the itemset appears in transactions")
        print("• CONFIDENCE: Probability of buying B given A was bought")
        print("• LIFT: How much more likely B is bought when A is bought (vs random)")
        print("• CONVICTION: Measure of rule dependence")

        print(f"\n🏆 TOP 20 ASSOCIATION RULES ('IF A, THEN B' FORMAT):")
        print("=" * 120)
        print(
            f"{'Rule ID':<8} {'Antecedent (A)':<25} {'Consequent (B)':<25} {'Support':<8} {'Confidence':<10} {'Lift':<8}")
        print("=" * 120)

        for i, (idx, rule) in enumerate(self.rules.head(20).iterrows(), 1):
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            support = rule['support']
            confidence = rule['confidence']
            lift = rule['lift']

            ant_display = antecedent[:23] + ".." if len(antecedent) > 25 else antecedent
            cons_display = consequent[:23] + ".." if len(consequent) > 25 else consequent

            print(f"Rule {i:<3} {ant_display:<25} {cons_display:<25} {support:<8.3f} {confidence:<10.3f} {lift:<8.3f}")


        print(f"\n🔍 DETAILED RULE INTERPRETATIONS:")
        print("=" * 80)

        for i, (idx, rule) in enumerate(self.rules.head(10).iterrows(), 1):
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))

            print(f"\nRule #{i}: IF customer buys [{antecedent}] THEN they will buy [{consequent}]")
            print(
                f"   📊 Confidence: {rule['confidence'] * 100:.1f}% - This rule is correct {rule['confidence'] * 100:.1f}% of the time")
            print(
                f"   📈 Support: {rule['support'] * 100:.2f}% - This combination occurs in {rule['support'] * 100:.2f}% of all transactions")
            print(
                f"   🚀 Lift: {rule['lift']:.2f} - Buying {antecedent} makes buying {consequent} {rule['lift']:.2f}x more likely")

            if rule['lift'] > 2:
                print(f"   💡 STRONG RELATIONSHIP: Very high likelihood of purchase together")
            elif rule['lift'] > 1.5:
                print(f"   ✨ GOOD RELATIONSHIP: Above average likelihood of purchase together")
            elif rule['lift'] > 1:
                print(f"   👍 POSITIVE RELATIONSHIP: Slightly more likely to be purchased together")

        # Rule categories analysis
        print(f"\n📈 RULE QUALITY ANALYSIS:")
        print("-" * 50)


        high_conf = self.rules[self.rules['confidence'] > 0.5]
        print(f"🎯 High Confidence Rules (>50%): {len(high_conf)} rules")
        print(f"   → These are very reliable predictions")


        high_lift = self.rules[self.rules['lift'] > 2.0]
        print(f"🚀 High Lift Rules (>2.0): {len(high_lift)} rules")
        print(f"   → These show strong item associations")


        high_support = self.rules[self.rules['support'] > 0.05]
        print(f"📊 High Support Rules (>5%): {len(high_support)} rules")
        print(f"   → These occur frequently in transactions")

        # Perfect rules (if any)
        perfect_rules = self.rules[self.rules['confidence'] == 1.0]
        if len(perfect_rules) > 0:
            print(f"⭐ Perfect Rules (100% confidence): {len(perfect_rules)} rules")
            print(f"   → These items are ALWAYS bought together")

        print(f"\n✅ TASK 2 COMPLETED: Successfully generated {len(self.rules)} 'if A, then B' association rules")

        return True

    def visualize_association_rules(self, top_n=15, save_plot=True):

        if self.rules is None or len(self.rules) == 0:
            print("❌ No rules to visualize")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Association Rules Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Support vs Confidence
        scatter1 = axes[0, 0].scatter(self.rules['support'], self.rules['confidence'],
                                      c=self.rules['lift'], cmap='viridis', alpha=0.6, s=50)
        axes[0, 0].set_xlabel('Support')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].set_title('Support vs Confidence (colored by Lift)')
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
        cbar1.set_label('Lift')

        # Plot 2: Support vs Lift
        scatter2 = axes[0, 1].scatter(self.rules['support'], self.rules['lift'],
                                      c=self.rules['confidence'], cmap='plasma', alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Support')
        axes[0, 1].set_ylabel('Lift')
        axes[0, 1].set_title('Support vs Lift (colored by Confidence)')
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
        cbar2.set_label('Confidence')

        # Plot 3: Top rules by confidence
        top_rules = self.rules.head(top_n)
        rule_labels = [f"{', '.join(list(row['antecedents']))} → {', '.join(list(row['consequents']))}"
                       for idx, row in top_rules.iterrows()]

        y_pos = np.arange(len(rule_labels))
        axes[1, 0].barh(y_pos, top_rules['confidence'], alpha=0.7, color='lightcoral')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels([label[:35] + '...' if len(label) > 35 else label
                                    for label in rule_labels], fontsize=8)
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_title(f'Top {top_n} Rules by Confidence')

        # Plot 4: Top rules by lift
        axes[1, 1].barh(y_pos, top_rules['lift'], alpha=0.7, color='lightgreen')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([label[:35] + '...' if len(label) > 35 else label
                                    for label in rule_labels], fontsize=8)
        axes[1, 1].set_xlabel('Lift')
        axes[1, 1].set_title(f'Top {top_n} Rules by Lift')

        plt.tight_layout()

        if save_plot:
            output_dir = os.path.dirname(self.data_path)
            plt.savefig(os.path.join(output_dir, 'association_rules_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Rules visualization saved to {os.path.join(output_dir, 'association_rules_analysis.png')}")

        plt.show()

    def analyze_customer_behavior(self):

        print("\n" + "=" * 80)
        print("TASK 3: CUSTOMER BEHAVIOR ANALYSIS - PURCHASING HABITS & PREFERENCES")
        print("=" * 80)
        print("📋 OBJECTIVE: Gain insights into customer purchasing habits and preferences")
        print("🔍 METHOD: Analyze association rules to understand shopping patterns")

        if self.rules is None:
            print("❌ Please run generate_association_rules() first")
            return False

        print(f"✅ ANALYZING {len(self.rules)} association rules for customer insights...")

        # Customer habit categories
        print(f"\n📊 CUSTOMER PURCHASING HABIT CATEGORIES:")
        print("=" * 60)

        # 1. Strong habits (high confidence)
        high_confidence_rules = self.rules[self.rules['confidence'] > 0.5]
        print(f"🎯 STRONG PURCHASING HABITS (Confidence > 50%): {len(high_confidence_rules)} patterns")
        print(f"   → These represent reliable customer behavior patterns")

        # 2. Popular combinations (high lift)
        high_lift_rules = self.rules[self.rules['lift'] > 2.0]
        print(f"🚀 POPULAR ITEM COMBINATIONS (Lift > 2.0): {len(high_lift_rules)} patterns")
        print(f"   → These show items customers strongly associate together")

        # 3. Frequent behaviors (high support)
        high_support_rules = self.rules[self.rules['support'] > 0.03]
        print(f"📈 FREQUENT SHOPPING BEHAVIORS (Support > 3%): {len(high_support_rules)} patterns")
        print(f"   → These represent common customer purchasing patterns")

        # Most impactful customer behaviors
        impactful_rules = self.rules[
            (self.rules['confidence'] > 0.3) & (self.rules['lift'] > 1.5)
            ]

        print(f"\n🔍 DETAILED CUSTOMER BEHAVIOR INSIGHTS:")
        print("=" * 80)
        print(f"Analyzing {len(impactful_rules)} significant behavior patterns...")

        # Customer behavior patterns
        behavior_insights = []

        for idx, rule in impactful_rules.head(15).iterrows():
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))

            behavior_insights.append({
                'trigger_items': antecedent,
                'follow_up_items': consequent,
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'support': rule['support']
            })

            print(f"\n🛒 CUSTOMER BEHAVIOR PATTERN #{len(behavior_insights)}:")
            print(f"   When customers buy: {antecedent}")
            print(f"   They are {rule['confidence'] * 100:.1f}% likely to also buy: {consequent}")
            print(f"   This happens {rule['lift']:.2f}x more often than by chance")
            print(f"   Found in {rule['support'] * 100:.2f}% of all shopping trips")

            # Interpret the behavior
            if rule['confidence'] > 0.7:
                behavior_type = "VERY STRONG habit"
            elif rule['confidence'] > 0.5:
                behavior_type = "STRONG habit"
            elif rule['confidence'] > 0.3:
                behavior_type = "MODERATE habit"
            else:
                behavior_type = "WEAK tendency"

            print(f"   💡 INTERPRETATION: This represents a {behavior_type} in customer purchasing")

        # Shopping pattern categories
        print(f"\n🛍️ SHOPPING PATTERN ANALYSIS:")
        print("=" * 60)

        # Find complementary shopping patterns
        complementary_patterns = []
        substitute_patterns = []

        if len(high_lift_rules) > 0:
            print(f"\n1. 🤝 COMPLEMENTARY SHOPPING PATTERNS (Items bought together):")
            for idx, rule in high_lift_rules.head(10).iterrows():
                ant = ', '.join(list(rule['antecedents']))
                cons = ', '.join(list(rule['consequents']))
                complementary_patterns.append((ant, cons, rule['lift']))
                print(f"   • {ant} + {cons} (Lift: {rule['lift']:.2f})")

                # Explain why they're complementary
                if rule['lift'] > 3:
                    print(f"     → VERY STRONG complementary relationship")
                elif rule['lift'] > 2:
                    print(f"     → STRONG complementary relationship")
                else:
                    print(f"     → MODERATE complementary relationship")

        # Find potential substitute patterns (low lift but reasonable support)
        substitutes = self.rules[(self.rules['lift'] < 1.0) & (self.rules['support'] > 0.02)]
        if len(substitutes) > 0:
            print(f"\n2. 🔄 POTENTIAL SUBSTITUTE PATTERNS (Alternative choices):")
            for idx, rule in substitutes.head(5).iterrows():
                ant = ', '.join(list(rule['antecedents']))
                cons = ', '.join(list(rule['consequents']))
                substitute_patterns.append((ant, cons, rule['lift']))
                print(f"   • {ant} vs {cons} (Lift: {rule['lift']:.2f})")
                print(f"     → Customers might choose one OR the other")

        # Customer preference insights
        print(f"\n👥 CUSTOMER PREFERENCE INSIGHTS:")
        print("=" * 50)

        # Most predictable customers
        most_predictable = self.rules[self.rules['confidence'] > 0.6]
        if len(most_predictable) > 0:
            print(f"🎯 PREDICTABLE SHOPPING: {len(most_predictable)} highly predictable patterns found")
            print(f"   → Some customer segments have very consistent shopping habits")

        # Impulse buying patterns (high lift, low support)
        impulse_patterns = self.rules[(self.rules['lift'] > 2.5) & (self.rules['support'] < 0.05)]
        if len(impulse_patterns) > 0:
            print(f"⚡ IMPULSE BUYING: {len(impulse_patterns)} impulse purchase patterns detected")
            print(f"   → These items trigger spontaneous additional purchases")

        # Routine shopping patterns (high support, moderate confidence)
        routine_patterns = self.rules[(self.rules['support'] > 0.05) &
                                      (self.rules['confidence'] > 0.2) &
                                      (self.rules['confidence'] < 0.5)]
        if len(routine_patterns) > 0:
            print(f"🔄 ROUTINE SHOPPING: {len(routine_patterns)} routine purchase patterns found")
            print(f"   → These represent regular, habitual shopping behaviors")

        # Customer segmentation insights
        print(f"\n👨‍👩‍👧‍👦 CUSTOMER SEGMENTATION INSIGHTS:")
        print("-" * 40)

        # Analyze by item categories to infer customer types
        breakfast_rules = self.rules[
            self.rules.apply(lambda x: any('coffee' in str(item).lower() or
                                           'bread' in str(item).lower() or
                                           'milk' in str(item).lower()
                                           for item in list(x['antecedents']) + list(x['consequents'])), axis=1)
        ]

        if len(breakfast_rules) > 0:
            print(f"☕ BREAKFAST SHOPPERS: {len(breakfast_rules)} breakfast-related patterns")
            print(f"   → Customers who buy breakfast items show specific additional preferences")

        # Health-conscious patterns
        health_rules = self.rules[
            self.rules.apply(lambda x: any('fruit' in str(item).lower() or
                                           'vegetable' in str(item).lower() or
                                           'yogurt' in str(item).lower()
                                           for item in list(x['antecedents']) + list(x['consequents'])), axis=1)
        ]

        if len(health_rules) > 0:
            print(f"🥗 HEALTH-CONSCIOUS SHOPPERS: {len(health_rules)} health-related patterns")
            print(f"   → Customers buying healthy items have distinct purchasing behaviors")

        print(f"\n✅ TASK 3 COMPLETED: Successfully analyzed customer purchasing habits and preferences")
        print(f"📈 KEY FINDINGS SUMMARY:")
        print(f"   • {len(complementary_patterns)} complementary shopping patterns identified")
        print(f"   • {len(substitute_patterns)} substitute choice patterns found")
        print(f"   • {len(behavior_insights)} significant behavior patterns analyzed")
        print(f"   • Customer segmentation insights derived from purchase associations")

        return True

    def generate_recommendations(self):

        print("\n" + "=" * 80)
        print("TASK 4: BUSINESS RECOMMENDATIONS - ACTIONABLE STRATEGIES")
        print("=" * 80)
        print("📋 OBJECTIVE: Provide actionable business recommendations based on analysis")
        print("🎯 FOCUS: Convert market basket insights into practical business strategies")

        if self.rules is None:
            print("❌ Please run generate_association_rules() first")
            return False

        print(f"✅ GENERATING RECOMMENDATIONS from {len(self.rules)} association rules...")

        # Strategic recommendation categories
        recommendations = []

        # 1. CROSS-SELLING OPPORTUNITIES
        high_confidence = self.rules[self.rules['confidence'] > 0.4]
        if len(high_confidence) > 0:
            recommendations.append({
                'category': '🎯 CROSS-SELLING OPPORTUNITIES',
                'description': 'Leverage strong associations to increase basket size',
                'rules': high_confidence.head(8),
                'priority': 'HIGH'
            })

        # 2. PROMOTIONAL STRATEGIES
        high_lift = self.rules[self.rules['lift'] > 2.0]
        if len(high_lift) > 0:
            recommendations.append({
                'category': '🎉 PROMOTIONAL STRATEGIES',
                'description': 'Create targeted promotions based on strong item associations',
                'rules': high_lift.head(8),
                'priority': 'HIGH'
            })

        # 3. INVENTORY MANAGEMENT
        high_support = self.rules[self.rules['support'] > 0.05]
        if len(high_support) > 0:
            recommendations.append({
                'category': '📦 INVENTORY MANAGEMENT',
                'description': 'Optimize stock levels for frequently bought together items',
                'rules': high_support.head(8),
                'priority': 'MEDIUM'
            })

        # 4. STORE LAYOUT OPTIMIZATION
        layout_rules = self.rules[(self.rules['lift'] > 1.5) & (self.rules['confidence'] > 0.3)]
        if len(layout_rules) > 0:
            recommendations.append({
                'category': '🏪 STORE LAYOUT OPTIMIZATION',
                'description': 'Arrange products based on purchase associations',
                'rules': layout_rules.head(6),
                'priority': 'MEDIUM'
            })

        print(f"\n📊 RECOMMENDATION CATEGORIES GENERATED: {len(recommendations)}")
        print("=" * 80)

        # Display detailed recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['category']} (Priority: {rec['priority']})")
            print(f"   💡 STRATEGY: {rec['description']}")
            print(f"   📈 EVIDENCE: Based on {len(rec['rules'])} association rules")
            print("   🔑 SPECIFIC RECOMMENDATIONS:")

            for j, (idx, rule) in enumerate(rec['rules'].iterrows(), 1):
                ant = ', '.join(list(rule['antecedents']))
                cons = ', '.join(list(rule['consequents']))

                print(f"\n      Recommendation {i}.{j}:")
                print(f"      📌 Target: Customers who buy [{ant}]")
                print(f"      🎯 Action: Promote/suggest [{cons}]")
                print(f"      📊 Expected Success: {rule['confidence'] * 100:.1f}% confidence")
                print(f"      🚀 Impact: {rule['lift']:.2f}x more effective than random")
                print(f"      💰 Market Size: {rule['support'] * 100:.2f}% of customers")

                # Specific tactical recommendations
                if rec['category'].startswith('🎯'):  # Cross-selling
                    print(f"      💼 TACTICS:")
                    print(f"         • Bundle {ant} with {cons} at checkout")
                    print(f"         • Train staff to suggest {cons} when {ant} is purchased")
                    print(f"         • Create 'frequently bought together' displays")

                elif rec['category'].startswith('🎉'):  # Promotions
                    print(f"      💼 TACTICS:")
                    print(f"         • Offer discount on {cons} when buying {ant}")
                    print(f"         • Create combo deals: {ant} + {cons}")
                    print(f"         • Send targeted coupons for {cons} to {ant} buyers")

                elif rec['category'].startswith('📦'):  # Inventory
                    print(f"      💼 TACTICS:")
                    print(f"         • Synchronize stock levels of {ant} and {cons}")
                    print(f"         • Ensure both items are always available together")
                    print(f"         • Adjust reorder points based on association strength")

                elif rec['category'].startswith('🏪'):  # Layout
                    print(f"      💼 TACTICS:")
                    print(f"         • Place {ant} and {cons} in adjacent aisles")
                    print(f"         • Create end-cap displays featuring both items")
                    print(f"         • Use shelf placement to encourage joint purchases")

        # ADDITIONAL STRATEGIC RECOMMENDATIONS
        print(f"\n{len(recommendations) + 1}. 🎪 ADVANCED MARKETING STRATEGIES:")
        print("   💡 STRATEGY: Implement data-driven marketing campaigns")
        print("   🔑 SPECIFIC RECOMMENDATIONS:")

        # Email marketing
        print(f"\n      {len(recommendations) + 1}.1 PERSONALIZED EMAIL CAMPAIGNS:")
        top_rules = self.rules.head(5)
        for idx, rule in top_rules.iterrows():
            ant = ', '.join(list(rule['antecedents']))
            cons = ', '.join(list(rule['consequents']))
            print(f"         • Email customers who bought {ant}: 'You might also like {cons}'")
            print(f"           Success rate: {rule['confidence'] * 100:.1f}%")

        # Loyalty programs
        print(f"\n      {len(recommendations) + 1}.2 LOYALTY PROGRAM OPTIMIZATION:")
        print(f"         • Reward points for purchasing associated items together")
        print(f"         • Create tier benefits based on basket completion patterns")
        print(f"         • Offer exclusive access to complementary product launches")

        # Digital recommendations
        print(f"\n      {len(recommendations) + 1}.3 DIGITAL RECOMMENDATION ENGINE:")
        print(f"         • Implement 'Customers who bought X also bought Y' on website")
        print(f"         • Mobile app push notifications for complementary items")
        print(f"         • AI-powered shopping list suggestions")

        print(f"\n{len(recommendations) + 2}. 📱 TECHNOLOGY IMPLEMENTATION:")
        print("   💡 STRATEGY: Leverage technology for automatic recommendations")
        print("   🔑 SPECIFIC RECOMMENDATIONS:")
        print("      • Point-of-sale system integration for real-time suggestions")
        print("      • Mobile app with AI-powered shopping assistant")
        print("      • Smart shopping cart technology")
        print("      • Beacon technology for location-based promotions")

        print(f"\n{len(recommendations) + 3}. 📈 PERFORMANCE MONITORING:")
        print("   💡 STRATEGY: Measure and optimize recommendation effectiveness")
        print("   🔑 SPECIFIC RECOMMENDATIONS:")
        print("      • Track basket size increase after implementing recommendations")
        print("      • Monitor cross-selling success rates")
        print("      • A/B test different promotional strategies")
        print("      • Regular market basket analysis updates (monthly/quarterly)")

        # ROI PROJECTIONS
        print(f"\n💰 EXPECTED RETURN ON INVESTMENT (ROI) PROJECTIONS:")
        print("=" * 60)

        # Calculate potential impact
        total_transactions = len(self.transactions)
        avg_basket_value = 50  # Assumed average basket value

        print(f"📊 IMPACT CALCULATIONS (Based on {total_transactions:,} transactions):")

        # Cross-selling impact
        if len(high_confidence) > 0:
            cross_sell_potential = high_confidence['support'].sum() * total_transactions
            revenue_increase = cross_sell_potential * avg_basket_value * 0.2  # 20% basket increase
            print(f"   🎯 Cross-selling: Potential {cross_sell_potential:.0f} additional sales")
            print(f"      Estimated revenue increase: ${revenue_increase:,.2f}")

        # Promotional impact
        if len(high_lift) > 0:
            promo_reach = high_lift['support'].mean() * total_transactions
            promo_revenue = promo_reach * avg_basket_value * 0.15  # 15% increase from promotions
            print(f"   🎉 Promotions: Reach {promo_reach:.0f} customers per campaign")
            print(f"      Estimated campaign revenue: ${promo_revenue:,.2f}")

        # Final summary
        print(f"\n✅ TASK 4 COMPLETED: Generated comprehensive business recommendations")
        print(f"📋 RECOMMENDATIONS SUMMARY:")
        print(f"   • {len(recommendations)} major strategy categories")
        print(f"   • {sum(len(r['rules']) for r in recommendations)} specific action items")
        print(f"   • Technology implementation roadmap provided")
        print(f"   • ROI projections and performance monitoring framework included")
        print(f"   • All recommendations based on data-driven market basket insights")

        return True

    def export_results(self):
        """Export analysis results to CSV files"""
        output_dir = os.path.dirname(self.data_path)

        try:
            if self.frequent_itemsets is not None:
                # Prepare frequent itemsets for export
                frequent_export = self.frequent_itemsets.copy()
                frequent_export['itemsets'] = frequent_export['itemsets'].apply(
                    lambda x: ', '.join(list(x))
                )
                output_path = os.path.join(output_dir, 'frequent_itemsets_results.csv')
                frequent_export.to_csv(output_path, index=False)
                print(f"✓ Frequent itemsets exported to '{output_path}'")

            if self.rules is not None:
                # Prepare rules for export
                rules_export = self.rules.copy()
                rules_export['antecedents'] = rules_export['antecedents'].apply(
                    lambda x: ', '.join(list(x))
                )
                rules_export['consequents'] = rules_export['consequents'].apply(
                    lambda x: ', '.join(list(x))
                )
                output_path = os.path.join(output_dir, 'association_rules_results.csv')
                rules_export.to_csv(output_path, index=False)
                print(f"✓ Association rules exported to '{output_path}'")

            # Export summary statistics
            if self.transactions:
                summary_data = {
                    'Metric': [
                        'Total Transactions',
                        'Total Unique Items',
                        'Average Items per Transaction',
                        'Frequent Itemsets Found',
                        'Association Rules Generated',
                        'High Confidence Rules (>40%)',
                        'High Lift Rules (>2.0)'
                    ],
                    'Value': [
                        len(self.transactions),
                        len(self.df_encoded.columns) if self.df_encoded is not None else 0,
                        np.mean([len(t) for t in self.transactions]),
                        len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
                        len(self.rules) if self.rules is not None else 0,
                        len(self.rules[self.rules['confidence'] > 0.4]) if self.rules is not None else 0,
                        len(self.rules[self.rules['lift'] > 2.0]) if self.rules is not None else 0
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                output_path = os.path.join(output_dir, 'analysis_summary.csv')
                summary_df.to_csv(output_path, index=False)
                print(f"✓ Analysis summary exported to '{output_path}'")

            return True

        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return False

    def print_final_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "=" * 60)
        print("📊 FINAL ANALYSIS SUMMARY")
        print("=" * 60)

        if self.transactions:
            print(f"📋 Total transactions analyzed: {len(self.transactions):,}")
            print(f"📦 Unique items in dataset: {len(self.df_encoded.columns) if self.df_encoded is not None else 0}")
            print(f"📈 Average items per transaction: {np.mean([len(t) for t in self.transactions]):.2f}")

        if self.frequent_itemsets is not None:
            print(f"🔍 Frequent itemsets discovered: {len(self.frequent_itemsets)}")

        if self.rules is not None:
            print(f"📏 Association rules generated: {len(self.rules)}")
            print(f"🎯 High confidence rules (>40%): {len(self.rules[self.rules['confidence'] > 0.4])}")
            print(f"🚀 High lift rules (>2.0): {len(self.rules[self.rules['lift'] > 2.0])}")

            # Best rule
            best_rule = self.rules.iloc[0]
            ant = ', '.join(list(best_rule['antecedents']))
            cons = ', '.join(list(best_rule['consequents']))
            print(f"\n🏆 Best Rule: {ant} → {cons}")
            print(f"   Confidence: {best_rule['confidence'] * 100:.1f}%, Lift: {best_rule['lift']:.2f}")

        print(f"\n📁 Results saved to: {os.path.dirname(self.data_path)}")
        print("\n✅ Market Basket Analysis Complete!")

    def run_complete_analysis(self, min_support=0.01, min_confidence=0.1,
                              create_visualizations=True, export_results=True):
        """
        Run the complete market basket analysis pipeline covering all 4 quiz tasks

        Parameters:
        - min_support: Minimum support threshold for frequent itemsets
        - min_confidence: Minimum confidence threshold for association rules
        - create_visualizations: Whether to create and save plots
        - export_results: Whether to export results to CSV files
        """
        print("🛒 MARKET BASKET ANALYSIS - GROCERY STORE DATA")
        print("=" * 80)
        print("Course: MSDA9223 - Data Mining and Information Retrieval")
        print("Assignment: Application of Market Basket Analysis using Association Rules")
        print("=" * 80)

        # Quiz Task Overview
        print("📋 QUIZ TASK REQUIREMENTS:")
        print("   Task 1: Use Association to discover patterns (frequent itemsets)")
        print("   Task 2: Generate association rules ('if A, then B' format)")
        print("   Task 3: Understanding customer behavior and preferences")
        print("   Task 4: Draw actionable business recommendations")
        print("=" * 80)

        # Step 0: Data loading and preprocessing
        print("\n🔄 STEP 0: DATA LOADING AND PREPROCESSING")
        if not self.load_grocery_data():
            print("❌ ANALYSIS FAILED: Cannot proceed without data")
            return False

        # Basic data overview
        if not self.analyze_data_overview():
            print("❌ ANALYSIS FAILED: Cannot analyze data overview")
            return False

        # Create initial visualizations
        if create_visualizations:
            print("\n📊 Creating data overview visualizations...")
            self.visualize_top_items()

        # TASK 1: Pattern Discovery
        print("\n" + "🔍 EXECUTING TASK 1...")
        if not self.find_frequent_itemsets(min_support=min_support):
            print("❌ TASK 1 FAILED: Cannot find frequent itemsets")
            return False

        # TASK 2: Rule Generation
        print("\n" + "⚡ EXECUTING TASK 2...")
        if not self.generate_association_rules(min_threshold=min_confidence):
            print("❌ TASK 2 FAILED: Cannot generate association rules")
            return False

        # Create rule visualizations
        if create_visualizations:
            print("\n📊 Creating association rules visualizations...")
            self.visualize_association_rules()

        # TASK 3: Customer Behavior Analysis
        print("\n" + "👥 EXECUTING TASK 3...")
        if not self.analyze_customer_behavior():
            print("❌ TASK 3 FAILED: Cannot analyze customer behavior")
            return False

        # TASK 4: Business Recommendations
        print("\n" + "💼 EXECUTING TASK 4...")
        if not self.generate_recommendations():
            print("❌ TASK 4 FAILED: Cannot generate recommendations")
            return False

        # Export results
        if export_results:
            print("\n💾 EXPORTING ANALYSIS RESULTS...")
            self.export_results()

        # Final summary with task completion status
        self.print_final_summary()

        print("\n" + "=" * 80)
        print("✅ ALL QUIZ TASKS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("📋 TASK COMPLETION STATUS:")
        print("   ✅ Task 1: Pattern Discovery - COMPLETED")
        print("   ✅ Task 2: Association Rules Generation - COMPLETED")
        print("   ✅ Task 3: Customer Behavior Analysis - COMPLETED")
        print("   ✅ Task 4: Business Recommendations - COMPLETED")
        print("=" * 80)

        return True


def main():
    print("🎓 DATA MINING COURSE - MARKET BASKET ANALYSIS QUIZ")
    print("=" * 60)
    print("Student: [100902 Faustin Mbarute]")
    print("Course: MSDA9223 - Data Mining and Information Retrieval")
    print("Instructor: Dr. Pacifique Nizeyimana")
    print("Assignment: Application of Market Basket Analysis using Association Rules")
    print("=" * 60)

    # Configuration parameters
    DATA_PATH = '/home/nkubito/Data_Minig_Course/Data/groceries.csv'
    MIN_SUPPORT = 0.01
    MIN_CONFIDENCE = 0.1

    if len(sys.argv) > 1:
        try:
            MIN_SUPPORT = float(sys.argv[1])
            print(f"📊 Using custom minimum support: {MIN_SUPPORT}")
        except ValueError:
            print("⚠️ Invalid support value provided, using default: 0.01")

    if len(sys.argv) > 2:
        try:
            MIN_CONFIDENCE = float(sys.argv[2])
            print(f"📊 Using custom minimum confidence: {MIN_CONFIDENCE}")
        except ValueError:
            print("⚠️ Invalid confidence value provided, using default: 0.1")

    print(f"\n🔧 ANALYSIS PARAMETERS:")
    print(f"   Dataset: {DATA_PATH}")
    print(f"   Minimum Support: {MIN_SUPPORT}")
    print(f"   Minimum Confidence: {MIN_CONFIDENCE}")

    analyzer = MarketBasketAnalyzer(data_path=DATA_PATH)

    success = analyzer.run_complete_analysis(
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        create_visualizations=True,
        export_results=True
    )

    if success:
        print("\n🎉 QUIZ SUBMISSION READY!")
        print("=" * 60)
        print("📚 ALL QUIZ REQUIREMENTS FULFILLED:")
        print("✅ Task 1: Association rules used to discover patterns and frequent itemsets")
        print("   → Identified frequent itemsets using Apriori algorithm")
        print("   → Discovered associations between different grocery items")
        print("   → Pattern analysis by itemset size (1-itemsets, 2-itemsets, etc.)")

        print("\n✅ Task 2: Generated association rules describing item relationships")
        print("   → Created 'if A, then B' format rules with confidence metrics")
        print("   → Rules indicate combinations of items often purchased together")
        print("   → Detailed interpretation of rule metrics (support, confidence, lift)")

        print("\n✅ Task 3: Understanding customer behavior and purchasing preferences")
        print("   → Analyzed customer purchasing habits and patterns")
        print("   → Identified complementary and substitute product relationships")
        print("   → Customer segmentation insights derived from purchase associations")

        print("\n✅ Task 4: Drew comprehensive business recommendations")
        print("   → Cross-selling and promotional strategies")
        print("   → Inventory management and store layout optimization")
        print("   → Technology implementation and ROI projections")

        print(f"\n📁 Output files generated in: {os.path.dirname(DATA_PATH)}")
        print("   • frequent_itemsets_results.csv")
        print("   • association_rules_results.csv")
        print("   • analysis_summary.csv")
        print("   • top_items_frequency.png")
        print("   • association_rules_analysis.png")

        print("\n🏆 ASSIGNMENT COMPLETE - READY FOR SUBMISSION!")

    else:
        print("\n❌ ANALYSIS FAILED")
        print("Please check the error messages above and ensure:")
        print("• The groceries.csv file exists at the specified path")
        print("• All required Python packages are installed")
        print("• You have sufficient permissions to read/write files")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())