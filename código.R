# ===============================
# PACOTES
# ===============================
pacotes <- c(
  "dplyr", "readr", "ggplot2", "plotly",
  "forecast", "prophet", "ggfortify", "broom", "lubridate", "tsibble", "fable", "fabletools", "feasts"
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

###################3 dcmp do trucios:
df_nordeste <- dfs_regioes[["Nordeste"]]
df_nordeste <- df_nordeste %>%
  mutate(din_instante = as.Date(din_instante))
serie_nordeste <- df_nordeste %>%
  as_tsibble(index = din_instante)

dcmp <- serie_nordeste |>    # passageiros é do tipo tsibble
  model(decomposition = classical_decomposition(val_cargaenergiamwmed, type = "mult")) |>  
  components() 

serie_nordeste %>%   # passageiros é do tipo tsibble
  model(classical_decomposition(val_cargaenergiamwmed, type = "multiplicative")) %>% 
  components() %>%  # extraimos as componentes T_t, S_t e R_t
  autoplot()


# ===============================
# 7. REGRESSÃO LINEAR
# Certifique-se que a coluna de data está como Date
df_nordeste <- dfs_regioes[["Nordeste"]]
df_nordeste <- df_nordeste %>%
  mutate(din_instante = as.Date(din_instante))

serie_nordeste_tsbl <- df_nordeste %>%
  as_tsibble(index = din_instante)



modelo_tslm <- serie_nordeste_tsbl %>%
  model(tslm = TSLM(val_cargaenergiamwmed ~ trend() + fourier(K = 2)))

report(modelo_tslm)

modelo_tslm %>% gg_tsresiduals()  ## HORROROSO


# ===============================

# ===============================
# 8. SUAVIZAÇÃO EXPONENCIAL TRIPLA (HOLTS-WINTER)
df_nordeste <- dfs_regioes[["Nordeste"]]
df_nordeste <- df_nordeste %>%
  mutate(din_instante = as_date(din_instante)) %>%
  arrange(din_instante)  # garante ordem temporal

df_nordeste_tsbl <- df_nordeste %>%
  as_tsibble(index = din_instante, regular = TRUE)

# Ajusta o modelo ETS (Holt-Winters automático)
modelo_ETS <- df_nordeste_tsbl %>%
  model(HWA = ETS(val_cargaenergiamwmed ~ error("A") + trend("Ad") + season("A")),
        HWM = ETS(val_cargaenergiamwmed ~ error("A") + trend("Ad") + season("A")))

modelo_ETS %>% select(HWA) %>% report() # Holts-winter aditivo

modelo_ETS %>% select(HWM) %>% gg_tsresiduals() ## residuos autocorrelacionados

## HW_pred <- modelo_ETS %>% forecast(h = 60)

## HW_pred |> autoplot(df_nordeste_tsbl) + geom_line(aes(y = .fitted, color = .model), data = augment(modelo_ETS)) 

#########################


# ===============================


# 9. PREVISÃO COM PROPHET
# ===============================







