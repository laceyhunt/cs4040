import pandas as pd
import random
import copy
import math
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

def preprocess_data(demand, df):
   # Create profit column for each candy
   df["profit"] = ((1 + 1*(df["winpercent"] / 100)) -(1*df["pricepercent"]))*demand*(df["winpercent"]/100)
   overall_dict = {}
   list_of_records = df.to_dict("records")
   for record in list_of_records:
      overall_dict[record['competitorname']] = record
   overall_dict.pop('One dime')
   overall_dict.pop('One quarter')
   return overall_dict

class Line:
   def __init__(self, candy_options, candy_dict, already_candies=[], line_type=None):
      self.candy_dict = candy_dict.copy()
      # Candies that will be present on this line
      self.candy_list = []
      # Missed an s
      self.candy_units = 0
      self.candy_limit = 8  # Total magic - can parameterize later
      self.candy_options = candy_options.copy()
      if line_type:
         if line_type=="fruit":
            # remove non-fruit candy
            for candy in candy_options:
               if self.candy_dict[candy]["fruity"]==0:
                  self.candy_options.remove(candy)
         elif line_type=="chocolate":
            # remove fruit candy
            for candy in candy_options:
               # Makes options ONLY chocolate type...
               # if self.candy_dict[candy]["chocolate"]==0:
               # Makes candy options everything but fruit (includes the peanut butter, etc.)
               if self.candy_dict[candy]["fruity"]==1:
                  self.candy_options.remove(candy)
      else:
         print("Did not give a line type option...")
            
      self.random_init_candies(already_candies=already_candies)
      self.prev_candy = None
      self.new_candy = None
   def get_candy_units(self, candy_name):
      if self.candy_dict[candy_name]["pluribus"] == 1:
         units = 2
      else:
         units = 1
      return units
   
   def random_init_candies(self, already_candies=[]):
      tries = 0
      candy_options = self.candy_options.copy()
      # If the candy is already being made somewhere in our line, etc
      for candy in already_candies:
         candy_options.remove(candy)
      while self.candy_units < self.candy_limit and tries < 20:
         # Pick a candy
         tries += 1
         candy_choice = random.choice(candy_options)
         units = self.get_candy_units(candy_choice)
         total_units = self.candy_units + units
         if total_units <= self.candy_limit:
            # Add the units
            self.candy_units += units
            # Add the candy to the list
            self.candy_list.append(candy_choice)
            self.candy_options.remove(candy_choice)
            candy_options.remove(candy_choice)
            
   def create_new_candy_options(self, other_candies=[]):
      new_candy_options = self.candy_options.copy()
      if other_candies != None:
         for candy in other_candies:
            new_candy_options.remove(candy)
      return new_candy_options

   def replace_candy(self, other_candies=None):
      # Randomly pick a candy on our line
      old_candy_choice = random.choice(self.candy_list)
      old_units = self.get_candy_units(old_candy_choice)
      new_candy_options = self.create_new_candy_options(other_candies=other_candies)
      new_candy_choice = random.choice(new_candy_options)
      new_units = self.get_candy_units(new_candy_choice)
      return old_candy_choice, old_units, new_candy_choice, new_units
   
   def mutate_candy(self, other_candies=None):
      tries = 0
      old_candy_choice, old_units, new_candy_choice, new_units = self.replace_candy(other_candies=other_candies)
      while ((self.candy_units - old_units + new_units > self.candy_limit)) and tries < 20:
         tries += 1
         old_candy_choice, old_units, new_candy_choice, new_units = self.replace_candy(other_candies=other_candies)
      if (self.candy_units - old_units + new_units) <= self.candy_limit:
         # Release old candy back into options
         self.candy_options.append(old_candy_choice)
         self.candy_units = self.candy_units - old_units
         self.candy_list.remove(old_candy_choice)
         # Add the new candy
         self.candy_list.append(new_candy_choice)
         self.prev_candy = old_candy_choice
         self.candy_units = self.candy_units + new_units
         self.new_candy = new_candy_choice
         self.candy_options.remove(new_candy_choice)
            
   def return_candy_list(self):
      return self.candy_list.copy()
   
   def calc_total_candy_profit(self):
      total_profit = 0
      for candy_name in self.candy_list:
         total_profit += self.candy_dict[candy_name]["profit"]
      return total_profit
   
   def print_self(self):
      print(f"Candies: {self.candy_list}")
      print(f"Candy Units: {self.candy_units}")
      print(f"Line Profit {round(self.calc_total_candy_profit(), 2)}")
      
   def print_profit(self):
      print(f"Line Profit {round(self.calc_total_candy_profit(), 2)}")
      
   def __lt__(self, other):
      return self.calc_total_candy_profit() < other.calc_total_candy_profit()

class Factory:
   def __init__(self, candy_options, candy_dict, already_candies=[]):
      self.fruit_line = Line(candy_options=candy_options, candy_dict=candy_dict, line_type="fruit")
      self.chocolate_line = Line(candy_options=candy_options, candy_dict=candy_dict, line_type="chocolate")
            
   def create_new_candy_options(self, other_candies=[]):
      new_candy_options = self.candy_options.copy()
      if other_candies != None:
         for candy in other_candies:
            new_candy_options.remove(candy)
      return new_candy_options

   def replace_candy(self, other_candies=None):
      # Randomly pick a candy on our line
      old_candy_choice = random.choice(self.candy_list)
      old_units = self.get_candy_units(old_candy_choice)
      new_candy_options = self.create_new_candy_options(other_candies=other_candies)
      new_candy_choice = random.choice(new_candy_options)
      new_units = self.get_candy_units(new_candy_choice)
      return old_candy_choice, old_units, new_candy_choice, new_units
   
   def mutate_candy(self, other_candies=None):
      self.fruit_line.mutate_candy()
      self.chocolate_line.mutate_candy()
            
   def return_candy_list(self):
      collective_candy_list = self.fruit_line.return_candy_list()
      choc = self.chocolate_line.return_candy_list()
      for c in choc:
         collective_candy_list.append(c)
      return collective_candy_list
   
   def calc_total_candy_profit(self):
      total_profit = 0
      total_profit += self.fruit_line.calc_total_candy_profit()
      total_profit += self.chocolate_line.calc_total_candy_profit()
      return total_profit
   
   def print_self(self):
      print(f"Candies: \n\tFruit:{self.fruit_line.candy_list}\n\tChocolate:{self.chocolate_line.candy_list}")
      units = self.chocolate_line.candy_units + self.fruit_line.candy_units
      print(f"Candy Units: {units}")
      print(f"Factory Profit {round(self.calc_total_candy_profit(), 2)}")
      
   def print_profit(self):
      print(f"Factory Profit {round(self.calc_total_candy_profit(), 2)}")
      
   def __lt__(self, other):
      return self.calc_total_candy_profit() < other.calc_total_candy_profit()

class Population:
   def __init__(self, candy_options, candy_dict, members, top_members, mutation_rate=0.1):
      self.candy_options = candy_options.copy()
      self.candy_dict = candy_dict.copy()
      self.member_num = members
      self.top_members_num = top_members
      self.tournament_size = 4         # Another good potential prarameterization
      self.mutation_rate = mutation_rate
      # self.mutation_rate = 0.2
      self.members = []
      self.top_members = []
      # Init population
      for i in range (0, self.member_num):
         new_factory=Factory(candy_options, candy_dict)
         self.members.append(new_factory)
      self.members.sort(reverse=True)  # We can do this bc we have the Line.__lt__() function
      # Copy best members to top members list
      for i in range (0, self.top_members_num):
         self.top_members.append(self.copy_member(self.members[i]))
   
   def update_top_rules(self): # Updates top members
      self.members.sort(reverse=True)
      self.top_members = self.members[:self.top_members_num]
   
   def copy_member(self, og):
      # Init a new individual
      new_factory = Factory(self.candy_options, self.candy_dict)
      # Copy the candy list, options, and units from og
      # chocolate line...
      new_factory.chocolate_line.candy_list = og.chocolate_line.candy_list.copy()
      new_factory.chocolate_line.candy_options = og.chocolate_line.candy_options.copy()
      new_factory.chocolate_line.candy_units = og.chocolate_line.candy_units
      # fruit line...
      new_factory.fruit_line.candy_list = og.fruit_line.candy_list.copy()
      new_factory.fruit_line.candy_options = og.fruit_line.candy_options.copy()
      new_factory.fruit_line.candy_units = og.fruit_line.candy_units
      return new_factory
   
   def mutate(self):
      # Num pop members to mutate based on mutation rate
      mutation_num = math.floor(self.member_num * self.mutation_rate)
      # Sample of mutants
      to_mutate = random.sample(self.members, mutation_num)
      # Perform mutation
      for member in to_mutate:
         member.mutate_candy()
   
   def tournament_selection(self):
      selection_list = random.sample(self.members, self.tournament_size)
      selection_list.sort(reverse=True)
      winner = selection_list[0]
      return self.copy_member(winner)
   
   def new_generation(self):
      new_gen = []
      for i in range(0, self.member_num):
         new_gen.append(self.tournament_selection())
      self.members = new_gen
      self.members.sort(reverse=True)
      
   def run_generation(self):
      # self.update_top_rules()
      self.mutate()
      self.new_generation()
      self.update_top_rules()

   def print_top_members(self, num_members = None):
      if num_members == None:
         num_members = self.top_members_num
      self.top_members.sort(reverse=True)
      for i in range(0,num_members):
         self.top_members[i].print_self()
         
   def print_top_members_profit(self, num_members = None):
      if num_members == None:
         num_members = self.top_members_num
      self.top_members.sort(reverse=True)
      for i in range(0,num_members):
         self.top_members[i].print_profit()

def run_ga(pop_size=200, top_members_num=10, num_gens=100, mutation_rate=0.1):
   pop = Population(candy_options, candy_dict, pop_size, top_members_num,mutation_rate=mutation_rate)
   top_profits=[]
   for i in range(0,num_gens):
      print(f"Generation {i}")
      pop.print_top_members_profit(num_members=1)
      top_profits.append(pop.top_members[0].calc_total_candy_profit())
      pop.run_generation()
   print("-----ENDING-----")
   with open("output.txt", "a") as f:
      with redirect_stdout(f):
         print(f"Population size: {pop_size}, # Generations: {num_gens}, Mutation Rate: {mutation_rate}")
         pop.print_top_members(num_members=1)
         print()
   pop.print_top_members(num_members=1)
   return top_profits

with open("output.txt", "w") as f:
   with redirect_stdout(f):
      print("Results of various genetic algorithm iterations:")
demand = 796142
df = pd.read_csv("../../datasets/candy-data.csv")
candy_dict = preprocess_data(demand, df)
candy_options = list(candy_dict.keys())


top_profits1=run_ga(num_gens=100, mutation_rate=0.1, pop_size=100)
top_profits2=run_ga(num_gens=100, mutation_rate=0.2, pop_size=100)
top_profits3=run_ga(num_gens=100, mutation_rate=0.3, pop_size=100)
top_profits4=run_ga(num_gens=100, mutation_rate=0.4, pop_size=100)
# Plot top profit
plt.figure(figsize=(10,6))
plt.plot(top_profits1, label="0.1")
plt.plot(top_profits2, label="0.2")
plt.plot(top_profits3, label="0.3")
plt.plot(top_profits4, label="0.4")

plt.xlim(0,100)
plt.legend(title="Mutation Rate")
plt.xlabel("Generation")
plt.ylabel("Value ($ Million)")
plt.suptitle("Top Profit Over Time", fontsize=15)
plt.title(f"Population Size: 100", fontsize=10)
# plt.grid(True)
plt.savefig(f"plots/compare_mutrate.png")
plt.clf()


top_profits1=run_ga(num_gens=100, mutation_rate=0.3, pop_size=100)
top_profits2=run_ga(num_gens=100, mutation_rate=0.3, pop_size=200)
top_profits3=run_ga(num_gens=100, mutation_rate=0.3, pop_size=300)
top_profits4=run_ga(num_gens=100, mutation_rate=0.3, pop_size=400)
# Plot top profit
plt.figure(figsize=(10,6))
plt.plot(top_profits1, label="100")
plt.plot(top_profits2, label="200")
plt.plot(top_profits3, label="300")
plt.plot(top_profits4, label="400")

plt.xlim(0,100)
plt.legend(title="Population Size")
plt.xlabel("Generation")
plt.ylabel("Value ($ Million)")
plt.suptitle("Top Profit Over Time", fontsize=15)
plt.title(f"Mutation Rate: 0.3", fontsize=10)
# plt.grid(True)
plt.savefig(f"plots/compare_popsize.png")
plt.clf()

top_profits1=run_ga(num_gens=100, mutation_rate=0.3, pop_size=100)
top_profits2=run_ga(num_gens=200, mutation_rate=0.3, pop_size=100)
top_profits3=run_ga(num_gens=300, mutation_rate=0.3, pop_size=100)
top_profits4=run_ga(num_gens=400, mutation_rate=0.3, pop_size=100)
# Plot top profit
plt.figure(figsize=(10,6))
plt.plot(top_profits1, label="100")
plt.plot(top_profits2, label="200")
plt.plot(top_profits3, label="300")
plt.plot(top_profits4, label="400")

plt.xlim(0,400)
plt.legend(title="# Generations")
plt.xlabel("Generation")
plt.ylabel("Value ($ Million)")
plt.suptitle("Top Profit Over Time", fontsize=15)
plt.title(f"Mutation Rate: 0.3, Population Size: 100", fontsize=10)
# plt.grid(True)
plt.savefig(f"plots/compare_numgens.png")
plt.clf()
