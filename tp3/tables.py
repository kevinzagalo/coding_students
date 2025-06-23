import pandas as pd

french_alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
letter_frequencies = pd.DataFrame(
    index=french_alphabet,
    data=dict(
        frequency=[8.38, 1.11, 3.60, 3.47, 16.69, 1.12, 0.86, 0.90, 7.05, 1.16, 0.05, 4.75, 3.19, 6.89, 6.06, 3.22, 1.04, 6.56, 8.14, 6.54, 6.50, 1.74, 0.07, 0.37, 0.27, 0.18]
    )
)