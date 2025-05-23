import pandas as pd
import numpy as np

def dropNulls(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Elimina las filas donde la columna especificada tiene valores nulos.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        column (str): Nombre de la columna a procesar
        
    Returns:
        pd.DataFrame: DataFrame con las filas nulas eliminadas
    """
    return df.dropna(subset=[column])

def imputMedian(df: pd.DataFrame, column: str, use_zero: bool = False) -> pd.DataFrame:
    """
    Imputa valores nulos con la mediana o cero de la columna especificada.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        column (str): Nombre de la columna a procesar
        use_zero (bool): Si True, imputa con cero. Si False, imputa con la mediana
        
    Returns:
        pd.DataFrame: DataFrame con los valores nulos imputados
    """
    df_copy = df.copy()
    if use_zero:
        df_copy[column] = df_copy[column].fillna(0)
    else:
        median_value = df_copy[column].median()
        df_copy[column] = df_copy[column].fillna(median_value)
    return df_copy

def imputMedianMidpoint(df: pd.DataFrame, column: str, use_zero: bool = False) -> pd.DataFrame:
    """
    Calcula el punto medio entre el mínimo y máximo, normaliza al punto medio,
    calcula la mediana y la usa para imputar valores nulos.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        column (str): Nombre de la columna a procesar
        use_zero (bool): Si True, imputa con cero. Si False, imputa con la mediana normalizada
        
    Returns:
        pd.DataFrame: DataFrame con los valores nulos imputados
    """
    df_copy = df.copy()
    
    if use_zero:
        df_copy[column] = df_copy[column].fillna(0)
    else:
        # Calcular mínimo y máximo excluyendo nulos
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        midpoint = (min_val + max_val) / 2
        
        # Normalizar al punto medio
        normalized_median = (df_copy[column].median() - min_val) / (max_val - min_val) * (max_val - min_val) + min_val
        
        # Imputar valores nulos
        df_copy[column] = df_copy[column].fillna(normalized_median)
    
    return df_copy

def imputMedianBinary(df: pd.DataFrame, column: str, use_zero: bool = False) -> pd.DataFrame:
    """
    Convierte la columna a binaria (0/1) basado en la mediana y luego imputa valores nulos.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        column (str): Nombre de la columna a procesar
        use_zero (bool): Si True, imputa con cero. Si False, imputa con la mediana binarizada
        
    Returns:
        pd.DataFrame: DataFrame con la columna binarizada y valores nulos imputados
    """
    df_copy = df.copy()
    
    # Calcular mediana excluyendo nulos
    median_value = df_copy[column].median()
    
    # Binarizar la columna
    df_copy[column] = (df_copy[column] > median_value).astype(int)
    
    # Imputar valores nulos
    if use_zero:
        df_copy[column] = df_copy[column].fillna(0)
    else:
        # La mediana binarizada será 0 o 1
        binary_median = int(df_copy[column].median())
        df_copy[column] = df_copy[column].fillna(binary_median)
    
    return df_copy

def imputCategoricalMode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Imputa valores nulos con el valor más frecuente (moda) de la columna categórica.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        column (str): Nombre de la columna categórica a procesar
        
    Returns:
        pd.DataFrame: DataFrame con los valores nulos imputados con la moda
    """
    df_copy = df.copy()
    mode_value = df_copy[column].mode()[0]
    df_copy[column] = df_copy[column].fillna(mode_value)
    return df_copy

def imputCategoricalUnknown(df: pd.DataFrame, column: str, create_flag: bool = True) -> pd.DataFrame:
    """
    Imputa valores nulos con 'Unknown' y opcionalmente crea una columna flag binaria.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        column (str): Nombre de la columna categórica a procesar
        create_flag (bool): Si True, crea una columna flag binaria
        
    Returns:
        pd.DataFrame: DataFrame con los valores nulos imputados y opcionalmente una columna flag
    """
    df_copy = df.copy()
    # Crear flag binario antes de imputar
    if create_flag:
        flag_column = f"{column}_is_null"
        df_copy[flag_column] = df_copy[column].isnull().astype(int)
    
    # Imputar con 'Unknown'
    df_copy[column] = df_copy[column].fillna('Unknown')
    return df_copy

def dropHighNullColumns(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Elimina columnas con un porcentaje de nulos mayor al umbral especificado.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        threshold (float): Umbral de porcentaje de nulos (0-1)
        
    Returns:
        pd.DataFrame: DataFrame sin las columnas con alta tasa de nulos
    """
    df_copy = df.copy()
    null_percentages = df_copy.isnull().mean()
    columns_to_drop = null_percentages[null_percentages > threshold].index
    return df_copy.drop(columns=columns_to_drop)

def dropConstantColumns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas que tienen un único valor (constantes).
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        
    Returns:
        pd.DataFrame: DataFrame sin las columnas constantes
    """
    df_copy = df.copy()
    constant_columns = [col for col in df_copy.columns if df_copy[col].nunique() == 1]
    return df_copy.drop(columns=constant_columns)

def applyNullHandling(df: pd.DataFrame, column: str, recommendation: str, use_zero: bool = False) -> pd.DataFrame:
    """
    Aplica el manejo de nulos según la recomendación especificada.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        column (str): Nombre de la columna a procesar
        recommendation (str): Tipo de manejo de nulos a aplicar
        use_zero (bool): Si True, usa cero para imputación cuando sea aplicable
        
    Returns:
        pd.DataFrame: DataFrame con los valores nulos manejados según la recomendación
    """
    df_copy = df.copy()
    
    # Manejo de variables numéricas
    if recommendation == "Drop rows if missing":
        return dropNulls(df, column)
    elif recommendation == "Impute with median":
        return imputMedian(df, column, use_zero=False)  # Siempre usar mediana
    elif recommendation == "Impute with 0":
        return imputMedian(df, column, use_zero=True)   # Siempre usar cero
    elif recommendation == "Impute with 1":
        df_copy[column] = df_copy[column].fillna(1)     # Imputar con uno
        return df_copy
    elif recommendation == "Impute with median + create midpoint":
        return imputMedianMidpoint(df, column, use_zero)
    elif recommendation == "Impute with median + create categories" or recommendation == "Impute with 0 + create categories":
        return imputMedian(df, column, use_zero)         
    elif recommendation == "Impute with 0 + create binary flag" or recommendation == "Impute with median + create binary flag":
        return imputMedianBinary(df, column, use_zero)
    
    # Manejo de variables categóricas
    elif recommendation == "Imputar con valor más frecuente":
        return imputCategoricalMode(df, column)
    elif recommendation == "Imputar con 'Unknown' + crear flag binario":
        return imputCategoricalUnknown(df, column, create_flag=True)
    elif recommendation == "Eliminar columna - Alta tasa de nulos":
        return dropHighNullColumns(df, threshold=0.7)
    elif recommendation == "No - Valor constante":
        return dropConstantColumns(df)
    else:
        print(f"Recomendación no implementada: {recommendation}")
        return df

def map_recommendation_to_function(recommendation: str) -> str:
    """
    Mapea las recomendaciones del CSV a las funciones de manejo de nulos.
    
    Args:
        recommendation (str): Recomendación del CSV
        
    Returns:
        str: Nombre de la función correspondiente
    """
    recommendation_map = {
        "No aplica - Sin nulos": None,
        "Imputar con 'Unknown' + crear flag binario": "Imputar con 'Unknown' + crear flag binario",
        "Imputar con valor más frecuente": "Imputar con valor más frecuente",
        "Imputar con mediana": "Impute with median",
        "Imputar con 0": "Impute with 0",
        "Imputar con 1": "Impute with 1",
        "Imputar con mediana + crear flag binario": "Impute with median + create binary flag",
        "Eliminar columna - Alta tasa de nulos": "Eliminar columna - Alta tasa de nulos",
        "No - Valor constante": "No - Valor constante",
        "Imputar con 'N' + crear flag binario": "Imputar con 'Unknown' + crear flag binario",  # Caso especial para hardship_flag
        "Imputar con mediana + crear categorías": "Impute with median + create categories",
        "Imputar con 0 + crear categorías": "Impute with 0 + create categories"
    }
    return recommendation_map.get(recommendation, None)

def processCategoricalColumnsFromCSV(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """
    Procesa las columnas categóricas según las recomendaciones especificadas en un archivo CSV.
    Omite las columnas que tienen "No aplica - Sin nulos" en su recomendación.
    Solo elimina las columnas que tienen la recomendación "Eliminar columna - Alta tasa de nulos".
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        csv_path (str): Ruta al archivo CSV con las recomendaciones
        
    Returns:
        pd.DataFrame: DataFrame con las columnas categóricas procesadas
    """
    # Preprocesar revol_util si existe
    df_copy = preprocess_revol_util(df)
    
    # Leer el CSV de recomendaciones
    recommendations_df = pd.read_csv(csv_path, sep=';')
    
    # Obtener las columnas que debemos eliminar
    columnas_a_eliminar = recommendations_df[
        (recommendations_df['conservar_columna'].str.startswith('No'))
    ]['column_name'].tolist()
    
    # Filtrar solo las columnas que necesitan procesamiento (excluyendo las que se eliminarán)
    recommendations_df = recommendations_df[
        (recommendations_df['conservar_columna'].str.startswith('Sí')) & 
        (recommendations_df['recomendacion_nulos'] != "No aplica - Sin nulos") &
        (recommendations_df['recomendacion_nulos'] != "Eliminar columna - Alta tasa de nulos")
    ]
    
    # Crear diccionario de recomendaciones
    recommendations_dict = {}
    for _, row in recommendations_df.iterrows():
        print(f"Procesando columna: {row['column_name']} - Recomendación: {row['recomendacion_nulos']}")
        column = row['column_name']
        recommendation = row['recomendacion_nulos']
        mapped_recommendation = map_recommendation_to_function(recommendation)
        
        if mapped_recommendation is not None:
            recommendations_dict[column] = mapped_recommendation
    
    # Procesar las columnas en orden específico
    # 1. Primero procesar columnas con valor constante
    for column, recommendation in recommendations_dict.items():
        if recommendation == "No - Valor constante":
            df_copy = applyNullHandling(df_copy, column, recommendation)
    
    # 2. Luego procesar las columnas restantes
    for column, recommendation in recommendations_dict.items():
        if recommendation != "No - Valor constante":
            df_copy = applyNullHandling(df_copy, column, recommendation)
    
    # 3. Eliminar solo las columnas que tienen recomendación de eliminación
    if columnas_a_eliminar:
        print(f"\nEliminando columnas con alta tasa de nulos: {', '.join(columnas_a_eliminar)}")
        df_copy = df_copy.drop(columns=columnas_a_eliminar)
    
    return df_copy

def preprocess_revol_util(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    if 'revol_util' in df_copy.columns:
        # Eliminar el símbolo '%' y convertir a float
        df_copy['revol_util'] = df_copy['revol_util'].str.rstrip('%').astype('float') / 100.0
    return df_copy