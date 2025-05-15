# ===============================
# PACOTES
# ===============================
pacotes <- c(
  "dplyr", "readr", "ggplot2", "plotly",
  "forecast", "gridExtra", "prophet", "ggfortify", "broom", "lubridate", "tsibble", "fable", "fabletools", "feasts", "gridExtra"
)
# Instala pacotes faltantes
pacotes_faltando <- pacotes[!(pacotes %in% installed.packages()[, "Package"])]
if(length(pacotes_faltando)) install.packages(pacotes_faltando)

# Carrega pacotes
lapply(pacotes, library, character.only = TRUE)

# ===============================
# 1. CARREGAR DADOS
# ===============================
url <- "https://raw.githubusercontent.com/abibernardo/modelagem_hierarquica_temporal/refs/heads/main/dados_me607.csv"
df_total <- read_csv(url) |> arrange(din_instante)

# Série total do Brasil
# ===============================
df_brasil <- df_total |> 
  group_by(din_instante) |> 
  summarise(val_cargaenergiamwmed = sum(val_cargaenergiamwmed), .groups = "drop")

# Dados por região
# ===============================
regioes <- unique(df_total$nom_subsistema)
dfs_regioes <- lapply(regioes, function(reg) df_total |> filter(nom_subsistema == reg))
names(dfs_regioes) <- regioes
df_nordeste <- dfs_regioes[["Nordeste"]]
df_norte <- dfs_regioes[["Norte"]]
df_sul <- dfs_regioes[["Sul"]]
df_sudeste_centroeste <- dfs_regioes[["Sudeste/Centro-Oeste"]]


# ===============================
# 2. FUNÇÕES DE MODELAGEM E DIAGNÓSTICO
# ===============================

# Função de diagnóstico de resíduos (estilo gg_tsresiduals)
analisar_residuos <- function(residuos_df, titulo = "") {
  p1 <- ggplot(residuos_df, aes(x = data, y = residuo)) +
    geom_line() + geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = paste(titulo, "- Resíduos no tempo"), x = "Data", y = "Resíduo")
  p2 <- ggplot(residuos_df, aes(x = residuo)) +
    geom_histogram(bins = 30) +
    labs(title = paste(titulo, "- Histograma dos resíduos"), x = "Resíduo", y = "Frequência")
  p3 <- autoplot(Acf(residuos_df$residuo, plot = FALSE)) +
    labs(title = paste(titulo, "- ACF dos resíduos"))
  grid.arrange(p1, p2, p3, ncol = 1)
}

# 2.1 Modelo SARIMA
ajustar_sarima <- function(df_regiao) {
  df_ts <- df_regiao %>% mutate(din_instante = as.Date(din_instante)) %>%
    as_tsibble(index = din_instante)
  modelo <- df_ts %>%
    model(SARIMA = ARIMA(val_cargaenergiamwmed ~ pdq(0:2,0:1,0:2) + PDQ(0:2,0:1,0:2)))
  print(report(modelo))
  fc <- forecast(modelo, h = 30)
  print(autoplot(fc, df_ts) + labs(title = "SARIMA Previsão", y = "MW médios", x = "Data"))
  # resíduos
  res <- modelo %>% augment() %>% select(din_instante, .innov) %>%
    rename(data = din_instante, residuo = .innov)
  analisar_residuos(res, titulo = "SARIMA")
}

# 2.2 Modelo Prophet
# 2.2 Modelo Prophet
ajustar_prophet <- function(df_regiao) {
  df_p <- df_regiao %>%
    transmute(ds = as.Date(din_instante), y = val_cargaenergiamwmed)
  
  modelo <- prophet(df_p,
                    yearly.seasonality = TRUE,
                    weekly.seasonality = FALSE,
                    daily.seasonality = FALSE)
  
  # gera futuros e previsão
  future <- make_future_dataframe(modelo, periods = 30)
  fc <- predict(modelo, future) %>%
    mutate(ds = as.Date(ds))    # <--- força Date nos forecasts
  
  # gráfico real vs fitted
  p <- ggplot() +
    geom_line(data = df_p, aes(x = ds, y = y), alpha = 0.5) +
    geom_line(data = fc, aes(x = ds, y = yhat), color = "blue") +
    labs(title = "Prophet Previsão", x = "Data", y = "MW médios")
  print(p)
  
  # resíduos
  df_res <- df_p %>%
    left_join(fc %>% select(ds, yhat), by = "ds") %>%
    mutate(residuo = y - yhat) %>%
    rename(data = ds)
  analisar_residuos(df_res, titulo = "Prophet")
}

# 2.3 Modelo ETS (Holt-Winters)
ajustar_ets <- function(df_regiao) {
  df_ts <- df_regiao %>% mutate(din_instante = as.Date(din_instante)) %>%
    arrange(din_instante) %>% as_tsibble(index = din_instante)
  modelo <- df_ts %>% model(ETS = ETS(val_cargaenergiamwmed ~ error("A") + trend("Ad") + season("A")))
  print(report(modelo))
  fc <- forecast(modelo, h = 30)
  print(autoplot(fc, df_ts) + labs(title = "ETS Previsão", y = "MW médios", x = "Data"))
  # resíduos
  res <- modelo %>% augment() %>% select(din_instante, .innov) %>%
    rename(data = din_instante, residuo = .innov)
  analisar_residuos(res, titulo = "ETS")
}

# 2.4 Modelo TSLM (Trend + Fourier)
ajustar_tslm <- function(df_regiao, K = 2) {
  df_ts <- df_regiao %>% mutate(din_instante = as.Date(din_instante)) %>% as_tsibble(index = din_instante)
  modelo <- df_ts %>% model(TSLM = TSLM(val_cargaenergiamwmed ~ trend() + fourier(K = K)))
  print(report(modelo))
  fc <- forecast(modelo, h = 30)
  print(autoplot(fc, df_ts) + labs(title = "TSLM Previsão", y = "MW médios", x = "Data"))
  # resíduos com fabletools
  res <- modelo %>% augment() %>% select(din_instante, .innov) %>% rename(data = din_instante, residuo = .innov)
  analisar_residuos(res, titulo = "TSLM")
}

# 2.5 Modelo SNAIVE
ajustar_snaive <- function(df_regiao) {
  df_ts <- df_regiao %>% mutate(din_instante = as.Date(din_instante)) %>%
    arrange(din_instante) %>% as_tsibble(index = din_instante)
  modelo <- df_ts %>% model(snaive = SNAIVE(val_cargaenergiamwmed ~ lag(7)))
  print(report(modelo))
  fc <- forecast(modelo, h = 30)
  print(autoplot(fc, df_ts) + labs(title = "SNAIVE Previsão", y = "MW médios", x = "Data"))
  # resíduos
  res <- modelo %>% augment() %>% select(din_instante, .innov) %>%
    rename(data = din_instante, residuo = .innov)
  analisar_residuos(res, titulo = "SNAIVE")
}

# ===============================
# 3. APLICAR FUNÇÕES EM CADA REGIÃO
# ===============================
# Região Nordeste
ajustar_sarima(df_nordeste)
ajustar_prophet(df_nordeste)
ajustar_ets(df_nordeste)
ajustar_tslm(df_nordeste, K = 2)
ajustar_snaive(df_nordeste)

# Região Norte
ajustar_sarima(df_norte)
ajustar_prophet(df_norte)
ajustar_ets(df_norte)
ajustar_tslm(df_norte, K = 2)
ajustar_snaive(df_norte)

# Região Sul
ajustar_sarima(df_sul)
ajustar_prophet(df_sul)
ajustar_ets(df_sul)
ajustar_tslm(df_sul, K = 2)
ajustar_snaive(df_sul)

# Região Sudeste + Centro-Oeste
ajustar_sarima(df_sudeste_centroeste)
ajustar_prophet(df_sudeste_centroeste)
ajustar_ets(df_sudeste_centroeste)
ajustar_tslm(df_sudeste_centroeste, K = 2)
ajustar_snaive(df_sudeste_centroeste)

# Brasil (total)
ajustar_sarima(df_brasil)
ajustar_prophet(df_brasil)
ajustar_ets(df_brasil)
ajustar_tslm(df_brasil, K = 2)
ajustar_snaive(df_brasil)


# ===============================
#### 4. MODELO HIERÁRQUICO
# ===============================

#### 4.1 Manipulação de dados e ajustes SARIMAs
df_hierarquia <- df_total %>%
  mutate(
    din_instante = as.Date(din_instante),
    pais = "Brasil"
  ) %>%
  as_tsibble(index = din_instante, key = c(pais, nom_subsistema)) %>%
  aggregate_key(pais / nom_subsistema, val_cargaenergiamwmed = sum(val_cargaenergiamwmed))

modelos_regionais <- df_hierarquia %>%
  model(sarima = ARIMA(val_cargaenergiamwmed))

##### 4.2 Ajustar modelos SARIMA para cada região Reconciliar com métodos hierárquicos (incluindo bottom-up)
modelos_reconciliados <- modelos_regionais %>%
  reconcile(
    bu        = bottom_up(sarima),
    ols       = min_trace(sarima, method = "ols"),
    var       = min_trace(sarima, method = "wls_var"),
    stru      = min_trace(sarima, method = "wls_struct"),
    mint_shr  = min_trace(sarima, method = "mint_shrink")
  )

##### 4.3 Predição hierárquica


# 1. (ex: bottom-up) (bu)
previsao_brasil <- modelos_reconciliados %>%
  forecast(h = 120)  # ou ols, var, etc.
previsao_brasil <- previsao_brasil %>%
  filter(pais == "Brasil", nom_subsistema == "<aggregated>", .model == "bu")


# ===============================
#### 5. COMPARAÇÃO MODELO DIRETO X MODELO HIERÁRQUICO
# ===============================

#### 5.1 Diretamente:
df_brasil_ts <- df_brasil %>%
  mutate(din_instante = as.Date(din_instante)) %>%
  as_tsibble(index = din_instante)

sarima_brasil <- df_brasil_ts %>%
  model(
    auto_sarima = ARIMA(val_cargaenergiamwmed)
  )

# Análise de resíduos

analisar_residuos <- function(residuos_df, titulo = "") {
  p1 <- ggplot(residuos_df, aes(x = data, y = residuo)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = paste(titulo, "- Resíduos no tempo"),
         x = "Data", y = "Resíduo")
  
  p2 <- ggplot(residuos_df, aes(x = residuo)) +
    geom_histogram(bins = 30) +
    labs(title = paste(titulo, "- Histograma dos resíduos"),
         x = "Resíduo", y = "Frequência")
  
  p3 <- autoplot( Acf(residuos_df$residuo, plot = FALSE) ) +
    labs(title = paste(titulo, "- ACF dos resíduos"))
  
  grid.arrange(p1, p2, p3, ncol = 1)
}


residuos_tsibble <- residuals(sarima_brasil)
residuos_tsibble <- residuos_tsibble %>%
  filter(.model == "auto_sarima")  # ou o nome que você deu no model()
residuos_df <- residuos_tsibble %>%
  as_tibble() %>%
  select(data = din_instante, residuo = .resid)

analisar_residuos(residuos_df, titulo = "SARIMA Brasil")

################## Fazer a previsão direta para h = 120 dias
previsao_direta <- sarima_brasil %>%
  forecast(h = 120)

### 5.2 COMPARAÇÃO

previsao_direta_df <- as.data.frame(previsao_direta)
previsao_brasil_df <- as.data.frame(previsao_brasil)

previsoes_comparadas <- bind_rows(
  previsao_direta_df %>%
    mutate(modelo = "Direto") %>%
    select(din_instante, .mean, modelo),
  previsao_brasil_df %>%
    mutate(modelo = "Hierárquico") %>%
    select(din_instante, .mean, modelo)
)

# Plota as duas séries de predições
ggplot(previsoes_comparadas, aes(x = din_instante, y = .mean, color = modelo)) +
  geom_line(size = 1) +
  labs(
    title = "Comparação das previsões: Direto vs Hierárquico",
    x = "Data",
    y = "Previsão de carga (MW méd)",
    color = "Modelo"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )



#### Dados fora da amostra
url2 <- "https://raw.githubusercontent.com/abibernardo/modelagem_hierarquica_temporal/main/CARGA_ENERGIA_2025%20(1).csv"
df_novo <- readr::read_csv2(url2) |> arrange(din_instante)
df_novo <- df_novo |> 
  group_by(din_instante) |> 
  summarise(val_cargaenergiamwmed = sum(val_cargaenergiamwmed), .groups = "drop")
df_novo <- df_novo %>%
  mutate(din_instante = as.Date(din_instante)) %>%
  arrange(din_instante)
df_novo <- df_novo[1:120, ]
df_novo <- df_novo %>%
  as_tsibble(index = din_instante)
 


accuracy(previsao_brasil, df_novo)
accuracy(previsao_direta, df_novo)

