# True-Random
`True-Random` is a small python library made to generate numbers & doing random operations **securely & async**. It's made for caases when you really want randomness, not just usual `random` stuff.

---

## Features
- Async-Friendly, so it won't block anything.
- Common random operations included:
	- `randint(a, b)` ~ get a random integer between A & B.
   	- `randrange(start, stop, step)` ~ gives a random number in a given range.
   	- `random()` ~ floating-point number between 0 & 1.
   	- `choice(seq)` ~ pick a random element from a sequence.
   	- `shuffle(list)` ~ shuffle a list securely.
  
   	- /!\ `d_shuffle(indices, password)` ~ deterministic shuffle based on a password.

> Note: `d_shuffle` is kinda project-specific and not really secure.

---

## Installation
Just include `true_random.py` in your project.
Works with python 3.10+.
