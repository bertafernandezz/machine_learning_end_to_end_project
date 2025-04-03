import pandas as pd


class BasePreprocessing:
    def __init__(self):
        self.df_out = None

    def preprocess(self, df: pd.DataFrame):
        self.df_out = df.copy()
        # eliminamos los espacios y los dígitos de la ubicación
        self.df_out.loc[:, "Standard_Location"] = (
            self.df_out["Location"]
            .str.lower()
            .str.replace(" ", "")
            .str.replace(r"\d+", "", regex=True)
            .str.replace("sector", "")
        )
        # calculamos el precio por metro cuadrado
        self.df_out["price_per_foot^2"] = self.df_out["Price"] / self.df_out["Area"]
        return self.df_out


class BaseModel:
    def __init__(self):
        self.df = None
        self.df_summary = None
        self.df_summary_per_city = None

    def train(self, df: pd.DataFrame):
        self.df = df.copy()

        # calculamos el precio por metro cuadrado por ciudad y ubicación estandarizada
        self.df_summary = self.df.groupby(["city", "Standard_Location"])[
            ["price_per_foot^2"]
        ].mean()

        self.df_summary_per_city = self.df.groupby(["city"])[
            ["price_per_foot^2"]
        ].mean()

    def predict(self, df: pd.DataFrame):
        df_out = df.copy()
        city_location_dict = self.df_summary.to_dict()["price_per_foot^2"]
        city_average_dict = self.df_summary_per_city.to_dict()["price_per_foot^2"]

        def calculate_price(x):
            city_average = city_average_dict[x["city"]]
            p_p_f = city_location_dict.get(
                (x["city"], x["Standard_Location"]), city_average
            )
            return p_p_f * x["Area"]

        result = df_out.apply(func=calculate_price, axis=1)
        return result.values
