import time

def print_text(text, indent=0, border=None):
    space = '  '
    to_print = f'{space*indent}{text}'

    if border is not None:
        to_print = f'{space*indent}{border*len(text)}\n' \
                   f'{to_print}\n' \
                   f'{space*indent}{border*len(text)}\n'

    print(to_print)

def print_time(task, start_time, indent=1):
    end_time = time.time()
    space = '  '

    print(f'{space*indent}Time taken to {task}: {end_time - start_time:.2f} seconds')
