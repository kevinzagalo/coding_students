
if __name__ == "__main__":
    import numpy as np
    import galois
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sb
    from __init__ import *

    # build qr encoder
    encoder = qrcode(
        version=1,
        mode='byte',
        level='l',
        mask=0
    )
    print(encoder)

    #
    area = encoder.data_area
    plt.imshow(area)
    