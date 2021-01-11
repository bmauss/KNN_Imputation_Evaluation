def cat_codes(df, columns):
    """
    Input: Data frame and list of columns
    Output: Columns converted to categories and assigned cat_codes
    """
    for i in columns:
        df[i] = df[i].astype('category')
        df[i] = df[i].cat.codes