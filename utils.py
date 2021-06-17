import pandas as pd
from classify import SplitDataFrame, TwoIndependentSamples


# import seaborn as sns
# import matplotlib.pyplot as plt


def loading_data():
    """
    Opening and loading data
    path: the path to the file
    wb_name:the name of the file (.xlsx)
    """
    wb_name = input("Type the filename: ")
    gap = pd.read_excel(wb_name, sheet_name='Dataset', header=3)
    del gap['Unnamed: 0']
    del gap['Unnamed: 11']
    print(f"\nDataFrame columns:")
    for i in list(gap.columns):
        print(f"\t{i}")

    return gap


def slicing_dataframe():
    """Choosing to slice or not the data frame"""
    print("\nChoose from the list:")
    print("\t1-work with the entire dataframe.")
    print("\t2-divide the dataframe.\n")
    temp = int(input("Your choice 1 or 2: "))
    limit, column = 0, None
    if temp == 1:
        direction = 'entire'
        return direction, limit, column
    elif temp == 2:
        print("\nChoose from the list.")
        print("Be aware that you can only choose from the list above and only numerical")
        print("and choose either higher or lower\n")
        column = input("Type the column (among the columns): ")
        limit = int(input("Type the value limit (number): "))
        print("\n\t1-choose value higher than the limit.")
        print("\t2-choose value lower than the limit.\n")
        temp = int(input("Your choice 1 or 2: "))
        if temp == 1:
            direction = 'higher'
            return direction, limit, column
        elif temp == 2:
            direction = 'lower'
            return direction, limit, column


def parameters_input(df):
    """Inputting the data"""
    print("\nWorking on the dataframe")
    field = input("\nType the field: ")
    ref = input("\nType the reference: ")
    print(f"\nThe values in {field}:")
    for i in set(df[field]):
        print(f"\t{i}")

    cat = input("\nType the category of reference: ")
    
    return df, field, ref, cat


def get_hypothesis():
    """Get the Null hypothesis"""
    print("\nThis is the main step of the work, we have to formulate the Null hypothesis.")
    print("Then we return it as a value to be checked by our algorithm\n")
    print("--------------------------------EXAMPLES----------------------------------------")
    print("Example 1: we want to check if apples in NY are more expensive than apples in LA.\n")
    print("The null hypothesis will be:")
    print("H0: $\mu_{NY} - \mu_{LA} <= 0$")
    print("H1: $\mu_{NY} - \mu_{LA} > 0$\n")
    print("Example 2: we want to check if the grades of eng students are 4% lower than mngmt students.")
    print("The null hypothesis will be:")
    print("H0: $\mu_{eng} - \mu_{mngmt} <= .04$")
    print("H1: $\mu_{eng} - \mu_{mngmt} > .04$\n")
    print("Then the value <0> or <.04> will be the reference value.")
    print("--------------------------------------------------------------------------------\n")
    d0 = float(input("Type the reference value: "))

    return d0


def main_ops():
    """Loading all the attributes"""
    df = loading_data()
    direction, limit, column = slicing_dataframe()  # Slicing of not
    df, field, ref, cat = parameters_input(df)  # Loading data

    if direction == "entire":
        pass
    else:
        df = SplitDataFrame(df, column, limit, direction)
        df = df.return_dataframe()

    print(type(df))

    res = TwoIndependentSamples(df, field, ref, cat)  # Assigning them to the calls Classify
    print(f"\nSummary of {field}: \n{res.get_count()}")  # Data Summary
    d0 = get_hypothesis()  # Get the null hypothesis value
    overall = res.compare_categories()
    print(f"\nComparison of {field}: \n{overall}")  # Result of the comparison
    t_score, p_val = res.get_stats_values(overall, d0)
    print(f"\nFor {field}: \nT-score: {t_score:.3f}\np-value: {p_val:.3f}")  # T-score and p-value
