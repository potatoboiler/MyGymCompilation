import model

import signal
import sys

def signal_handler(sig, frame):

    sys.exit(0)

def __main__():
    # need to pass benchmarks so that this actually works, lol
    analysis = model.train()

    



if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    __main__()