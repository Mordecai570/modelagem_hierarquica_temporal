# ===============================
# PACOTES
# ===============================
pacotes <- c(
  "dplyr", "readr", "ggplot2", "plotly",
  "forecast", "prophet", "ggfortify"
)

# Instala os pacotes que ainda não estão instalados
pacotes_faltando <- pacotes[!(pacotes %in% installed.packages()[, "Package"])]
if(length(pacotes_faltando)) {
  install.packages(pacotes_faltando)
}

# Carrega os pacotes
lapply(pacotes, library, character.only = TRUE)

# ===============================
# 1. CARREGAR DADOS
# ===============================
url <- "https://raw.githubusercontent.com/abibernardo/modelagem_hierarquica_temporal/refs/heads/main/dados_me607.csv"
df_total <- read_csv(url)

# Ordenar por data
df_total <- df_total |> arrange(din_instante)

# ===============================
# 2. SÉRIE TOTAL DO BRASIL
# ===============================
df_brasil <- df_total |>
  group_by(din_instante) |>
  summarise(val_cargaenergiamwmed = sum(val_cargaenergiamwmed))

# ===============================
# 3. DADOS POR REGIÃO
# ===============================
regioes <- unique(df_total$nom_subsistema)

dfs_regioes <- lapply(regioes, function(reg) {
  df_total |> filter(nom_subsistema == reg)
})
names(dfs_regioes) <- regioes

# ===============================
# 4. ANÁLISE EXPLORATÓRIA
# ===============================
# Brasil
p_brasil <- plot_ly(df_brasil, x = ~din_instante, y = ~val_cargaenergiamwmed, type = 'scatter', mode = 'lines') |>
  layout(title = "Consumo de Energia no Brasil")
p_brasil

# Por região
p_regioes <- plot_ly(df_total, x = ~din_instante, y = ~val_cargaenergiamwmed,
                     color = ~nom_subsistema, type = 'scatter', mode = 'lines') |>
  layout(title = "Consumo de Energia por Região")
p_regioes

# ===============================
# 5. CORRELOGRAMA - Exemplo: Nordeste
# ===============================
df_nordeste <- dfs_regioes[["Nordeste"]]
acf(df_nordeste$val_cargaenergiamwmed, lag_max = 50) |> autoplot()

# ===============================
# 6. DECOMPOSIÇÃO ADITIVA
# ===============================
serie_nordeste <- ts(df_nordeste$val_cargaenergiamwmed, frequency = 365)
decomposicao <- decompose(serie_nordeste, type = "additive")
autoplot(decomposicao)


# ===============================
# 7. REGRESSÃO LINEAR
  
--- USAR DUMMIES PARA ESTAÇÕES
--- USAR SAZONALIDADE E TENDÊNCIA

# ===============================

# ===============================
# 8. SUAVIZAÇÃO EXPONENCIAL TRIPLA (HOLTS-WINTER)

--- POIS HÁ SAZONALIDADE E TENDÊNCIA

# ===============================


# 9. PREVISÃO COM PROPHET
# ===============================
df_prophet <- df_nordeste |>
  rename(ds = din_instante, y = val_cargaenergiamwmed)

m <- prophet(df_prophet)
future <- make_future_dataframe(m, periods = 365)
forecast <- predict(m, future)

# Plot previsão
plot_ly() |>
  add_lines(x = df_prophet$ds, y = df_prophet$y, name = "Original") |>
  add_lines(x = forecast$ds, y = forecast$yhat, name = "Previsão") |>
  add_lines(x = forecast$ds, y = forecast$yhat_upper, name = "Limite Sup.", line = list(dash = "dot")) |>
  add_lines(x = forecast$ds, y = forecast$yhat_lower, name = "Limite Inf.", line = list(dash = "dot")) |>
  layout(title = "Previsão com Prophet - NORDESTE")

# Tendência
plot_ly(x = forecast$ds, y = forecast$trend, type = 'scatter', mode = 'lines', name = 'Tendência') |>
  layout(title = "Tendência - NORDESTE")

# Resíduos
residuos <- df_prophet
residuos$yhat <- forecast$yhat[1:nrow(residuos)]
residuos$residuo <- residuos$y - residuos$yhat

plot_ly(residuos, x = ~ds, y = ~residuo, type = 'scatter', mode = 'lines+markers') |>
  layout(title = "Resíduos - NORDESTE")
