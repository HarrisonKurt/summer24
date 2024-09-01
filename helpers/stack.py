class Stack:
  def __init__(self):
    self.prices = []

  def len(self):
    return len(self.prices)

  def push(self, value):
    self.prices.append(value)

  def pop(self):
    v = self.prices[0]
    self.prices = self.prices[1:]
    return v
