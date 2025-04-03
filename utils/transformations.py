"""
Modulo para procesar y transformar datos.
Todas las clases tendrán al menos dos métodos:
   - fit() -> para ajustar los parámetros
   - transform() -> para transformar las features.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from skrub import GapEncoder
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.preprocessing import PolynomialFeatures


class ExtendedTransformation:

    def __init__(self, ge_components=50):
        self.imputer = SimpleImputer(strategy="median")
        self.ohEnconder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.gapEncoder = GapEncoder(n_components=ge_components)
        self.y_Transformer = QuantileTransformer()
        self.area_Transformer = QuantileTransformer()
        self.beds_Transformer = QuantileTransformer()
        self.polyfeatures = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        self.scaler_y = StandardScaler()
        self.scaler_area = StandardScaler()
        self.scalar_beds = StandardScaler()

    def fit(self, X, y):
        X_data = X.copy()
        y_data = y.copy()
        print("X shape: ", X.shape)
        self.bin_vars_columns = X.columns[4:]
        print("bin_vars_columns shape: ", self.bin_vars_columns.shape)

        # fit impute n beds
        self.beds_feaures = "No. of Bedrooms"
        self.imputer.fit(X_data[[self.beds_feaures]])
        X_data = X_data.replace({9: np.nan})
        X_data[self.bin_vars_columns] = X_data[self.bin_vars_columns].replace(
            {0: "NO", 1: "SI", np.nan: "NO_DISPONIBLE"}
        )
        # fit low_cardinality features wiht ohot encoding
        self.low_card_columns = ["city"] + self.bin_vars_columns.to_list()
        print("low_card_columns shape: ", len(self.low_card_columns))  
        self.ohEnconder.fit(X_data[self.low_card_columns])
        self.loc_feature = "Location"

        # fit high_cardinality features.
        self.gapEncoder.fit(X_data[self.loc_feature])

        self.area_feature = "Area"

        # fit Quantile transformation of numerical vars.
        self.y_Transformer.fit(y_data)
        self.area_Transformer.fit(X_data[[self.area_feature]])
        self.beds_Transformer.fit(X_data[[self.beds_feaures]])

        self.scaler_y.fit(self.y_Transformer.transform(y_data))
        self.scaler_area.fit(
            self.area_Transformer.transform(X_data[[self.area_feature]])
        )
        self.scalar_beds.fit(
            self.beds_Transformer.transform(X_data[[self.beds_feaures]])
        )

        # scale to standard

    def transform(self, X_data, y_data):
        X = X_data.copy()
        y = y_data.copy()

        # impute missing data
        X = X.replace({9: np.nan})
        X[self.bin_vars_columns] = X[self.bin_vars_columns].replace(
            {0: "NO", 1: "SI", np.nan: "NO_DISPONIBLE"}
        )
        X[self.beds_feaures] = self.imputer.transform(X[[self.beds_feaures]])
        print("X shape: ", X.shape)
        # transform categorical features.
        cat_low_card_tfed = self.ohEnconder.transform(X[self.low_card_columns])
        X_low_card = pd.DataFrame(
            data=cat_low_card_tfed,
            columns=self.ohEnconder.get_feature_names_out(),
            index=X.index,
        )
        print("X_low_card   shape: ", X_low_card.shape)

        X_high_card = self.gapEncoder.transform(X[self.loc_feature])
        print("X_high_card shape: ", X_high_card.shape)

        # transform numerical vars.
        y_transformed = self.y_Transformer.transform(y)
        area_normal = self.area_Transformer.transform(X[[self.area_feature]])
        beds_normal = self.beds_Transformer.transform(X[[self.beds_feaures]])

        y_scaled = self.scaler_y.transform(y_transformed)
        area_scaled = self.scaler_area.transform(area_normal)
        beds_scaled = self.scalar_beds.transform(beds_normal)

        X_num = pd.DataFrame(
            data={
                self.area_feature: area_scaled.flatten(),
                self.beds_feaures: beds_scaled.flatten(),
            },
            index=X.index,
        )
        features_to_cross = pd.concat([X_low_card,X_num], axis=1)
        self.polyfeatures.fit(features_to_cross)
        crossed_features = self.polyfeatures.transform(features_to_cross)

        X_crossed_features = pd.DataFrame(
            data=crossed_features,
            columns=self.polyfeatures.get_feature_names_out(),
            index=X.index,
        )
        print("X_crossed_features shape: ", X_crossed_features.shape)
        X_EXPANDED = pd.concat([X_num, X_low_card, X_high_card, X_crossed_features], axis=1)
        print("X_EXPANDED shape: ", X_EXPANDED.shape)
        return X_EXPANDED, y_scaled

    def inverse_transform(self, y_data):
        return self.y_Transformer.inverse_transform(
            self.scaler_y.inverse_transform(y_data)
        )


class SimpleTransformation:

    def fit(self, X_data, y_data):
        self.remove_column = "Location"
        self.impute_columns = list(
            set(X_data.columns.to_list()) - set([self.remove_column, "city"])
        )
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(X_data[self.impute_columns])

        self.ohEnconder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohEnconder.fit(X_data[["city"]])

    def transform(self, X_data, y_data):
        X = X_data.copy()
        y = y_data.copy()
        X = X.drop(columns=[self.remove_column])
        X[self.impute_columns] = self.imputer.transform(X[self.impute_columns])
        X_cat = pd.DataFrame(
            data=self.ohEnconder.transform(X_data[["city"]]),
            columns=self.ohEnconder.get_feature_names_out(),
            index=X.index,
        )
        X_final = pd.concat([X.drop(columns=["city"]), X_cat], axis=1)
        return X_final, y
