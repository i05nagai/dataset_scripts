

def enumerate_step(iterable, start=0, step=1):
    counter = start
    for i in iterable:
        yield (counter, i)
        counter += step


def to_list_of_list(iterable, size=1):
    while True:
        yield [next(iterable) for i in range(size)]
