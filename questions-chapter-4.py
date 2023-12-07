import torch
tensor = torch.arange(1, 10) # or list(range(1, 10))
tensor = tensor.view(3, 3)

bottom_two_rows = tensor[1:]

without_first_column = bottom_two_rows[:, -2:]

print(without_first_column)


