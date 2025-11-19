import numpy as np
import matplotlib.pyplot as plt

# Business Monte Carlo Simulation
# Modeling function - maps out inputs as outputs
def calc_profit(demand, cost_to_produce, sale_price):
   margin = sale_price - cost_to_produce
   profit = demand*margin
   return profit

def get_price(price_type):
   if price_type == "low":
      return 10
   elif price_type == "medium":
      return 12
   elif price_type == "high":
      return 15

def get_demand_distribution(price_type):
   if price_type == "low":
      mean = 700
      stdv = 50
   elif price_type == "medium":
      mean = 600
      stdv = 10
   elif price_type == "high":
      mean = 400
      stdv = 100
   return mean,stdv

# Make choice - factors
mean_cost = 6
stdv_cost = 0.5
market_size = 1000
num_samples = 1000
price_type = "high"

# Run simulation
price = get_price(price_type)
mean, stdv = get_demand_distribution(price_type)
# Run the Monte Carlo sampling
demand_samples = np.random.normal(mean, stdv, num_samples)
cost_samples = np.random.normal(mean_cost, stdv_cost, num_samples)
profit_list = []
for demand_num, cost_num in zip(demand_samples, cost_samples):
   profit_num = calc_profit(demand_num, cost_num, price)
   profit_list.append(profit_num)
   
print(f"Worst Case {min(profit_list)}")
print(f"Best Case {max(profit_list)}")
print(f"Avg Case {sum(profit_list)/len(profit_list)}")
# Plot it
plt.hist(profit_list)
plt.title(f"{price_type} Output Profit Distribution")
plt.savefig(f"output_dist{price_type}.png")
plt.clf()