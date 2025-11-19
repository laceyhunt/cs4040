import pandas as pd
import random
import copy
import math
import json


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
   def __init__(self, candy_options, candy_dict, already_candies=[]):
      self.candy_dict = candy_dict.copy()
      # Candies that will be present on this line
      self.candy_list = []
      # Missed an s
      self.candy_units = 0
      self.candy_limit = 8  # Total magic - can parameterize later
      self.candy_options = candy_options.copy()
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
            # Pop this candy off our options list
            # WATCH
            # if candy_choice in self.candy_options:
            # self.candy_options.remove(candy_choice)
            self.candy_options.remove(candy_choice)
            # ADD THIS LINE
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


class Population:
   def __init__(self, candy_options, candy_dict, members, top_members):
      self.candy_options = candy_options.copy()
      self.candy_dict = candy_dict.copy()
      self.member_num = members
      self.top_members_num = top_members
      self.tournament_size = 4         # Another good potential prarameterization
      self.mutation_rate = 0.1
      # self.mutation_rate = 0.2
      self.members = []
      self.top_members = []
      # Init population
      for i in range (0, self.member_num):
         new_line = Line(candy_options, candy_dict)
         self.members.append(new_line)
      self.members.sort(reverse=True)  # We can do this bc we have the Line.__lt__() function
      # Copy best members to top members list
      for i in range (0, self.top_members_num):
         self.top_members.append(self.copy_member(self.members[i]))
   
   def update_top_rules(self): # Updates top members
      self.members.sort(reverse=True)
      self.top_members = self.members[:self.top_members_num]
   
   def copy_member(self, og):
      # Init a new individual
      new_line = Line(self.candy_options, self.candy_dict)
      # Copy the candy list, options, and units from og
      new_line.candy_list = og.candy_list.copy()
      new_line.candy_options = og.candy_options.copy()
      new_line.candy_units = og.candy_units        # this is a number so no need to copy
      return new_line

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

demand = 796142
df = pd.read_csv("../datasets/candy-data.csv")
candy_dict = preprocess_data(demand, df)
# print(json.dumps(candy_dict, indent=4))
candy_options = list(candy_dict.keys())

pop_size = 200
top_members_num = 10
num_gens = 100
pop = Population(candy_options, candy_dict, pop_size, top_members_num)
for i in range(0,num_gens):
   print(f"Generation {i}")
   pop.print_top_members_profit(num_members=1)
   pop.run_generation()
   
print("-----ENDING-----")
pop.print_top_members(num_members=2)

# # pop.print_top_members(num_members=3)
# print("old:")
# pop.print_top_members_profit(num_members=3)
# # pop.new_generation()
# # pop.mutate()
# # pop.update_top_rules()
# pop.run_generation()
# print("new:")
# pop.print_top_members_profit(num_members=3)



# candy_line = Line(candy_options, candy_dict)
# # print(candy_line.return_candy_list())
# candy_line.print_self()
# # candy_line.print_profit()
# candy_line.mutate_candy()
# # print(candy_line.return_candy_list())
# candy_line.print_self()
# # candy_line.print_profit()
