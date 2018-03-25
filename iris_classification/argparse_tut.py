import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("square", type=int, help="display a square of a given number")
# args = parser.parse_args()
# print(args.square**2)

# parser = argparse.ArgumentParser()
# parser.add_argument("--verbosity", help="increase output verbosity")
# args = parser.parse_args()

# if args.verbosity:
#     print("verbosity turned on")

# parser = argparse.ArgumentParser()
# parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
# args = parser.parse_args()

# if args.verbose:
#     print("verbosity turned on")



parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("square", type=int, help="display a square of a given number")
args = parser.parse_args()
answer = args.square**2

if args.verbose:
    print("The square of {} equals {}".format(args.square, answer))
else:
    print(answer)
