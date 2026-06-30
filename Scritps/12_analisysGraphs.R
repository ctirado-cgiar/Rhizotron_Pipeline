# ============================================================
# 12_analisysGraphs.r - Rhizotron Pipeline
# Alliance of Bioversity International & CIAT
# Authors: Cristian Tirado-Murcia, Jorge Aragon, Jose Polania
# ============================================================
#
# USO:
# 1. Define CARPETA_BASE en la seccion de configuracion (ruta del ensayo)
# 2. Define N_ROW_PERFIL, N_COL_PERFIL, N_ROW_ESQUEMAS, N_COL_ESQUEMAS segun
#    el numero de genotipos de tu ensayo
# 3. Corre el script completo
# 4. Para exportar: usa export() en consola con el tamano que prefieras,
#    por ejemplo: ggsave("scatter_penetracion.png", width=10, height=7, dpi=300)

# --- LIBRERIAS ----------------------------------------------
library(tidyverse)
library(gt)
library(ggrepel)
library(colorspace)
library(patchwork)
library(scales)

# ============================================================
# CONFIGURACION DEL USUARIO
# ============================================================

# Carpeta base del ensayo. De aqui se derivan automaticamente:
#   <CARPETA_BASE>/11_rasgos/       -> donde estan los 3 CSV de entrada
#   <CARPETA_BASE>/analisisdatos/   -> donde se guardan las figuras
CARPETA_BASE <- "D:/OneDrive - CGIAR/Frijol/Procesamiento/Roots/Trial/BMFG_Validation/"

# Layout del panel de perfil de cobertura por zonas - un genotipo por celda
N_ROW_PERFIL <- 1
N_COL_PERFIL <- NULL   # NULL = se calcula automaticamente segun N_ROW_PERFIL

# Layout del panel de esquemas de arquitectura radicular
N_ROW_ESQUEMAS <- 2
N_COL_ESQUEMAS <- NULL   # NULL = se calcula automaticamente segun N_ROW_ESQUEMAS

# ============================================================
# RUTAS DERIVADAS (no editar)
# ============================================================
CARPETA_DATOS  <- file.path(CARPETA_BASE, "11_rasgos")
CARPETA_SALIDA <- file.path(CARPETA_BASE, "analisisdatos")

path_temporal  <- file.path(CARPETA_DATOS, "rasgos_temporales.csv")
path_desempeno <- file.path(CARPETA_DATOS, "rasgos_desempeno.csv")
path_angulos   <- file.path(CARPETA_DATOS, "angulos_laterales.csv")

dir.create(CARPETA_SALIDA, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(CARPETA_SALIDA, "esquemas_individuales"),
           recursive = TRUE, showWarnings = FALSE)

# --- CARGA DE DATOS -----------------------------------------
df_t   <- read_csv(path_temporal,  show_col_types = FALSE)
df_d   <- read_csv(path_desempeno, show_col_types = FALSE)
df_ang <- read_csv(path_angulos,   show_col_types = FALSE)

# --- PALETA DE COLORES DINAMICA ------------------------------
# Genera un color distinto por cada genotipo presente en los datos,
# sin importar cuantos sean ni como se llamen
genotipos_presentes <- sort(unique(df_d$genotipo))
colores <- setNames(
  hue_pal()(length(genotipos_presentes)),
  genotipos_presentes
)

# --- FUNCION SE ---------------------------------------------
se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))

# ============================================================
# PREPARACION DE DATOS
# ============================================================

# Tasa de exploracion del suelo (crecimiento del hull por dia)
df_exploracion <- df_t %>%
  arrange(plot, DAS) %>%
  group_by(plot, genotipo) %>%
  mutate(
    dt        = DAS - lag(DAS),
    hull_incr = area_hull_cm2 - lag(area_hull_cm2),
    tasa_hull = hull_incr / dt
  ) %>%
  filter(!is.na(tasa_hull)) %>%
  summarise(
    tasa_exploracion_mean = mean(tasa_hull, na.rm = TRUE),
    tasa_exploracion_se   = se(tasa_hull),
    .groups = "drop"
  )

# Resumen por genotipo (n replicas variable, no asume n=2)
df_gen <- df_d %>%
  group_by(genotipo) %>%
  summarise(
    n               = n(),
    prof_final_mean = mean(profundidad_final_cm,                na.rm = TRUE),
    prof_final_se   = se(profundidad_final_cm),
    tasa_prof_mean  = mean(tasa_profundizacion_promedio_cm_dia, na.rm = TRUE),
    tasa_prof_se    = se(tasa_profundizacion_promedio_cm_dia),
    n_lat_mean      = mean(n_laterales_final,                   na.rm = TRUE),
    n_lat_se        = se(n_laterales_final),
    dens_ram_mean   = mean(densidad_ramificacion_promedio,      na.rm = TRUE),
    dens_ram_se     = se(densidad_ramificacion_promedio),
    long_total_mean = mean(longitud_total_final_cm,             na.rm = TRUE),
    long_total_se   = se(longitud_total_final_cm),
    area_final_mean = mean(area_final_cm2,                      na.rm = TRUE),
    area_final_se   = se(area_final_cm2),
    tort_mean       = mean(tortuosidad_pivotante_promedio,      na.rm = TRUE),
    tort_se         = se(tortuosidad_pivotante_promedio),
    diam_max_mean   = mean(diametro_max_pivotante_cm,           na.rm = TRUE),
    diam_max_se     = se(diametro_max_pivotante_cm),
    .groups = "drop"
  ) %>%
  left_join(
    df_exploracion %>%
      group_by(genotipo) %>%
      summarise(
        explor_mean = mean(tasa_exploracion_mean, na.rm = TRUE),
        explor_se   = se(tasa_exploracion_mean),
        .groups = "drop"
      ),
    by = "genotipo"
  )

# Cobertura por zonas de profundidad (ultimo DAS disponible por plot)
# NOTA: este bloque asume nombres de columna "cobertura_<rango>cm_pct/cm2"
# generados por 11_globalTraitsExtraction.py. Si N_ZONAS cambia en el
# .env del pipeline, las zonas aqui se ajustan automaticamente porque
# se detectan por nombre de columna, no estan hardcodeadas en numero.
df_zonas <- df_t %>%
  group_by(plot, genotipo) %>%
  filter(DAS == max(DAS)) %>%
  ungroup() %>%
  group_by(genotipo) %>%
  summarise(
    across(starts_with("cobertura_") & ends_with("_pct"),
           list(mean = ~mean(., na.rm = TRUE),
                se   = ~se(.)),
           .names = "{.col}_{.fn}"),
    across(starts_with("cobertura_") & ends_with("_cm2"),
           list(mean = ~mean(., na.rm = TRUE)),
           .names = "{.col}_{.fn}"),
    .groups = "drop"
  )

# Detectar automaticamente las zonas presentes en los datos
nombres_zonas_pct <- names(df_zonas) %>%
  str_subset("_pct_mean$") %>%
  str_remove("cobertura_") %>%
  str_remove("_pct_mean$")

n_zonas_detectadas <- length(nombres_zonas_pct)

# Zonas en formato largo para graficar
df_zonas_long <- df_zonas %>%
  select(genotipo, ends_with("_pct_mean"), ends_with("_cm2_mean")) %>%
  pivot_longer(
    cols      = ends_with("_pct_mean"),
    names_to  = "zona",
    values_to = "pct_mean"
  ) %>%
  mutate(
    zona = str_remove(zona, "cobertura_"),
    zona = str_remove(zona, "_pct_mean$"),
    zona = str_replace_all(zona, "_", "-")
  ) %>%
  left_join(
    df_zonas %>%
      select(genotipo, ends_with("_cm2_mean")) %>%
      pivot_longer(
        cols      = ends_with("_cm2_mean"),
        names_to  = "zona_cm2",
        values_to = "cm2_mean"
      ) %>%
      mutate(
        zona_cm2 = str_remove(zona_cm2, "cobertura_"),
        zona_cm2 = str_remove(zona_cm2, "_cm2_mean$"),
        zona_cm2 = str_replace_all(zona_cm2, "_", "-")
      ) %>%
      rename(zona = zona_cm2),
    by = c("genotipo", "zona")
  ) %>%
  left_join(df_gen %>% select(genotipo, prof_final_mean),
            by = "genotipo") %>%
  arrange(desc(prof_final_mean)) %>%
  mutate(genotipo = factor(genotipo, levels = unique(genotipo)))

# Ordenar zonas por profundidad creciente (extrae el primer numero del rango)
orden_zonas <- df_zonas_long %>%
  distinct(zona) %>%
  mutate(zona_inicio = as.numeric(str_extract(zona, "^\\d+"))) %>%
  arrange(zona_inicio) %>%
  pull(zona)

df_zonas_long <- df_zonas_long %>%
  mutate(zona = factor(zona, levels = orden_zonas))

# Angulos: clasificacion por orden (1=principal, 2=secundario por longitud)
df_ang_ind <- df_ang %>%
  mutate(
    zona_inicio = as.numeric(str_extract(zona, "^\\d+")),
    zona_num    = match(zona_inicio, sort(unique(zona_inicio)))
  ) %>%
  group_by(genotipo, zona) %>%
  mutate(
    umbral_orden = quantile(longitud_cm, 0.5),
    orden        = ifelse(longitud_cm >= umbral_orden, "1", "2")
  ) %>%
  ungroup()

# Resumen de angulos por genotipo y zona (rango p15-p85)
df_ang_rango <- df_ang_ind %>%
  group_by(genotipo, zona, zona_num) %>%
  summarise(
    n_laterales   = n(),
    angulo_mean   = mean(angulo_deg,           na.rm = TRUE),
    angulo_min    = quantile(angulo_deg, 0.15, na.rm = TRUE),
    angulo_max    = quantile(angulo_deg, 0.85, na.rm = TRUE),
    longitud_mean = mean(longitud_cm,          na.rm = TRUE),
    .groups = "drop"
  )

cat(sprintf("\nGenotipos detectados (%d): %s\n",
            length(genotipos_presentes),
            paste(genotipos_presentes, collapse = ", ")))
cat(sprintf("Zonas de profundidad detectadas (%d): %s\n",
            n_zonas_detectadas,
            paste(orden_zonas, collapse = ", ")))

# ============================================================
# TABLA 3: Root system architecture performance by genotype
# ============================================================
tabla_x <- df_gen %>%
  arrange(desc(tasa_prof_mean)) %>%
  mutate(
    `Max depth (cm)`             = sprintf("%.1f ± %.1f", prof_final_mean, prof_final_se),
    `Penetration power (cm/day)` = sprintf("%.2f ± %.2f", tasa_prof_mean,  tasa_prof_se),
    `N laterals`                 = sprintf("%.0f ± %.0f", n_lat_mean,      n_lat_se),
    `Branch density (lat/cm)`    = sprintf("%.1f ± %.1f", dens_ram_mean,   dens_ram_se),
    `Total length (cm)`          = sprintf("%.0f ± %.0f", long_total_mean, long_total_se),
    `Max area (cm²)`             = sprintf("%.1f ± %.1f", area_final_mean, area_final_se),
    `Soil exploration (cm²/day)` = sprintf("%.1f ± %.1f", explor_mean,     explor_se)
  ) %>%
  select(
    Genotype                     = genotipo,
    `Max depth (cm)`,
    `Penetration power (cm/day)`,
    `N laterals`,
    `Branch density (lat/cm)`,
    `Total length (cm)`,
    `Max area (cm²)`,
    `Soil exploration (cm²/day)`
  )

tabla_gt <- tabla_x %>%
  gt() %>%
  tab_header(
    title    = "Root system architecture performance by genotype",
    subtitle = sprintf("Genotypes ordered by penetration power | Mean ± SE (n = %d)",
                       max(df_gen$n))
  ) %>%
  tab_spanner(
    label   = "Root penetration capacity",
    columns = c(`Max depth (cm)`, `Penetration power (cm/day)`)
  ) %>%
  tab_spanner(
    label   = "Root branching capacity",
    columns = c(`N laterals`, `Branch density (lat/cm)`)
  ) %>%
  tab_spanner(
    label   = "Overall system",
    columns = c(`Total length (cm)`, `Max area (cm²)`, `Soil exploration (cm²/day)`)
  ) %>%
  cols_align(align = "center", columns = everything()) %>%
  cols_align(align = "left",   columns = Genotype) %>%
  tab_style(
    style     = cell_text(weight = "bold"),
    locations = cells_column_spanners()
  ) %>%
  tab_style(
    style     = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_style(
    style     = cell_fill(color = "#156082", alpha = 0.4),
    locations = cells_body(rows = seq(1, nrow(tabla_x), 2))
  ) %>%
  tab_style(
    style     = cell_text(weight = "bold", size = 14),
    locations = cells_title(groups = "title")
  ) %>%
  tab_options(
    table.font.size                   = 12,
    heading.align                     = "left",
    column_labels.border.top.color    = "black",
    column_labels.border.bottom.color = "black",
    table_body.border.bottom.color    = "black"
  )

tabla_gt
gtsave(tabla_gt, file.path(CARPETA_SALIDA, "tabla3_performance_genotipo.png"))
cat("Guardada: tabla3_performance_genotipo.png\n")

# ============================================================
# Penetration scatter: max depth vs penetration power
# ============================================================
med_x <- median(df_gen$tasa_prof_mean)
med_y <- median(df_gen$prof_final_mean)
rango_x <- range(df_gen$tasa_prof_mean)
rango_y <- range(df_gen$prof_final_mean)

fig_scatter <- ggplot(df_gen,
                      aes(x = tasa_prof_mean, y = prof_final_mean, color = genotipo)) +
  
  geom_vline(xintercept = med_x,
             linetype = "dashed", color = "gray60", linewidth = 0.5) +
  geom_hline(yintercept = med_y,
             linetype = "dashed", color = "gray60", linewidth = 0.5) +
  
  geom_errorbar(
    aes(ymin = prof_final_mean - prof_final_se,
        ymax = prof_final_mean + prof_final_se),
    width = diff(rango_x) * 0.015, linewidth = 0.5, alpha = 0.5
  ) +
  geom_errorbarh(
    aes(xmin = tasa_prof_mean - tasa_prof_se,
        xmax = tasa_prof_mean + tasa_prof_se),
    height = diff(rango_y) * 0.02, linewidth = 0.5, alpha = 0.5
  ) +
  
  geom_point(aes(size = diam_max_mean), alpha = 0.85) +
  
  geom_label_repel(
    aes(label = genotipo),
    size = 3.5, fontface = "bold",
    box.padding = 0.5, show.legend = FALSE,
    color = "gray20", fill = "white", label.size = 0.2
  ) +
  
  annotate("text",
           x = rango_x[1] + diff(rango_x) * 0.02,
           y = rango_y[2] - diff(rango_y) * 0.03,
           label = "Slow & deep", size = 3,
           color = "gray50", fontface = "italic", hjust = 0) +
  annotate("text",
           x = rango_x[1] + diff(rango_x) * 0.02,
           y = rango_y[1] + diff(rango_y) * 0.03,
           label = "Slow & shallow", size = 3,
           color = "gray50", fontface = "italic", hjust = 0) +
  annotate("text",
           x = med_x + diff(rango_x) * 0.04,
           y = rango_y[2] - diff(rango_y) * 0.03,
           label = "Fast & deep", size = 3,
           color = "gray50", fontface = "italic", hjust = 0) +
  annotate("text",
           x = med_x + diff(rango_x) * 0.04,
           y = rango_y[1] + diff(rango_y) * 0.03,
           label = "Fast & shallow", size = 3,
           color = "gray50", fontface = "italic", hjust = 0) +
  
  scale_color_manual(values = colores) +
  scale_size_continuous(
    name   = "Max taproot\ndiameter (cm)",
    range  = c(3, 12),
    breaks = c(min(df_gen$diam_max_mean),
               median(df_gen$diam_max_mean),
               max(df_gen$diam_max_mean)),
    labels = function(x) sprintf("%.3f", x)
  ) +
  scale_x_continuous(expand = expansion(mult = 0.08)) +
  scale_y_continuous(expand = expansion(mult = 0.08)) +
  
  labs(
    title    = "Taproot penetration depth vs. penetration power",
    subtitle = "Point size = max taproot diameter (cm) | Error bars = SE | Dashed lines = medians",
    x        = "Penetration power (cm/day)",
    y        = "Max depth (cm)",
    color    = "Genotype"
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    legend.position  = "right",
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(color = "gray40", size = 9),
    plot.margin      = margin(10, 20, 10, 10)
  )

fig_scatter
ggsave(file.path(CARPETA_SALIDA, "scatter_penetracion.png"),
       fig_scatter, width = 9, height = 6.5, dpi = 300)
cat("Guardada: scatter_penetracion.png\n")
# Para exportar con otro tamano:
# ggsave(file.path(CARPETA_SALIDA, "scatter_penetracion.png"),
#        fig_scatter, width = 12, height = 8, dpi = 300)

# ============================================================
# Root coverage profile across soil depth zones
# ============================================================
generar_tonos <- function(color_base, n) {
  colorRampPalette(c(lighten(color_base, 0.5),
                     darken(color_base, 0.4)))(n)
}

df_perfil_centrado <- df_zonas_long %>%
  mutate(
    zona_idx    = as.numeric(zona),
    zona_limpia = str_remove(as.character(zona), "cm$"),
    zona_ini    = as.numeric(str_extract(zona_limpia, "^\\d+")),
    zona_fin    = as.numeric(str_extract(zona_limpia, "\\d+$")),
    prof_mid    = (zona_ini + zona_fin) / 2,
    alto_zona_real = zona_fin - zona_ini
  ) %>%
  mutate(
    gen_num = as.numeric(genotipo),
    xmin    = gen_num - pct_mean / 200,
    xmax    = gen_num + pct_mean / 200
  ) %>%
  rowwise() %>%
  mutate(
    color_fill = generar_tonos(colores[genotipo], n_zonas_detectadas)[zona_idx]
  ) %>%
  ungroup()

# Etiquetas de zona para el eje Y (ordenadas de superficial a profunda)
breaks_zona <- df_perfil_centrado %>%
  distinct(zona, prof_mid) %>%
  arrange(prof_mid)

fig_perfil <- ggplot(df_perfil_centrado) +
  
  geom_rect(
    aes(xmin = xmin, xmax = xmax,
        ymin = prof_mid - alto_zona_real/2, ymax = prof_mid + alto_zona_real/2,
        fill = color_fill),
    color = "white", linewidth = 0.2, alpha = 0.95
  ) +
  
  geom_vline(
    aes(xintercept = as.numeric(genotipo)),
    color = "gray80", linewidth = 0.3, linetype = "dotted"
  ) +
  
  geom_text(
    aes(x     = as.numeric(genotipo),
        y     = prof_mid,
        label = ifelse(pct_mean > 5, sprintf("%.0f%%", pct_mean), "")),
    size = 2.5, color = "white", fontface = "bold", angle = 90
  ) +
  
  scale_fill_identity() +
  
  scale_x_continuous(
    breaks = seq_along(levels(df_perfil_centrado$genotipo)),
    labels = levels(df_perfil_centrado$genotipo),
    expand = c(0.05, 0.05)
  ) +
  
  scale_y_reverse(
    breaks = breaks_zona$prof_mid,
    labels = breaks_zona$zona,
    expand = c(0, 0)
  ) +
  
  labs(
    title    = "Root area distribution profile by genotype",
    subtitle = "Bar width = % of total root area | Light = shallow | Dark = deep | Ordered by max depth",
    x        = "Genotype",
    y        = "Soil depth (cm)"
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3,
                                      linetype = "dotted"),
    axis.text.x        = element_text(face = "bold", size = 11),
    axis.text.y        = element_text(size = 10),
    plot.title         = element_text(face = "bold", size = 13),
    plot.subtitle      = element_text(color = "gray40", size = 9),
    legend.position    = "none",
    plot.margin        = margin(10, 20, 10, 10)
  )

# Si el usuario quiere mas de 1 fila, se usa facet_wrap con grupos de genotipos
if (N_ROW_PERFIL > 1) {
  n_gen <- length(genotipos_presentes)
  n_col_calc <- ifelse(is.null(N_COL_PERFIL),
                       ceiling(n_gen / N_ROW_PERFIL), N_COL_PERFIL)
  grupo_genotipo <- setNames(
    ceiling(seq_along(levels(df_perfil_centrado$genotipo)) / n_col_calc),
    levels(df_perfil_centrado$genotipo)
  )
  df_perfil_centrado$fila_panel <- grupo_genotipo[as.character(df_perfil_centrado$genotipo)]
  
  fig_perfil <- fig_perfil + facet_wrap(~fila_panel, scales = "free_x", ncol = 1)
}

fig_perfil
ggsave(file.path(CARPETA_SALIDA, "perfil_cobertura.png"),
       fig_perfil, width = max(9, length(genotipos_presentes) * 1.3),
       height = 6.5 * N_ROW_PERFIL, dpi = 300)
cat("Guardada: perfil_cobertura.png\n")
# Para exportar con otro tamano:
# ggsave(file.path(CARPETA_SALIDA, "perfil_cobertura.png"),
#        fig_perfil, width = 14, height = 8, dpi = 300)

# ============================================================
# Root system architecture schemes
# ============================================================
# IMPORTANTE: la sinuosidad del pivotante en estos esquemas es una
# representacion SIMULADA, no la trayectoria real de ninguna raiz.
# La magnitud de la sinuosidad esta calibrada al valor de tortuosidad
# medido (tort_mean) por genotipo, pero la forma especifica de la
# curva es generada aleatoriamente. Sirve para comunicar diferencias
# relativas entre genotipos, no para representar la forma exacta
# de un pivotante individual.

# Detectar columnas de zona de cobertura automaticamente
zonas_cols_pct <- names(df_zonas) %>% str_subset("_pct_mean$")

ANCHO_GRAFICO <- 5.6

# Profundidad total del rizotron y alto de cada zona, derivados de los
# nombres reales de columna (ej. "cobertura_0_10cm_pct_mean" -> 0 a 10).
# Esto soporta cualquier tamaño de zona (2cm, 10cm, 20cm, etc.) sin
# necesidad de hardcodear el rango total.
limites_zonas_reales <- zonas_cols_pct %>%
  str_remove("cobertura_") %>%
  str_remove("_pct_mean$") %>%
  str_remove("cm$") %>%
  str_split("_") %>%
  map(as.numeric)

PROF_TOTAL_RIZOTRON <- max(map_dbl(limites_zonas_reales, ~.x[2]))
ALTO_ZONA_GLOBAL     <- PROF_TOTAL_RIZOTRON / length(zonas_cols_pct)

gen_esquema_panel <- function(gen_nombre, gen_data, ang_data, zonas_cols,
                              mostrar_eje     = FALSE,
                              mostrar_eje_der = FALSE) {
  prof      <- gen_data$prof_final_mean
  tort      <- gen_data$tort_mean
  zonas_pct <- as.numeric(gen_data[1, zonas_cols])
  n_zonas   <- length(zonas_cols)
  alto_zona <- ALTO_ZONA_GLOBAL
  zonas_lim <- lapply(seq_len(n_zonas), function(i)
    c((i-1)*alto_zona, i*alto_zona))
  
  gen_x <- 0
  ratio <- PROF_TOTAL_RIZOTRON / ANCHO_GRAFICO
  
  # Pivotante sinuoso (simulado, calibrado por tortuosidad medida)
  n_pts   <- 60
  y_pts   <- seq(0, prof, length.out = n_pts)
  desv    <- (tort - 1) * 3.5
  set.seed(which(genotipos_presentes == gen_nombre))
  x_noise <- cumsum(c(0, rnorm(n_pts - 1, 0, desv * 0.06)))
  x_noise <- x_noise - mean(x_noise)
  x_pts   <- gen_x + x_noise
  piv     <- data.frame(x = x_pts, y = y_pts)
  
  lats <- data.frame()
  
  for (i in seq_along(zonas_lim)) {
    rng        <- zonas_lim[[i]]
    if (rng[1] >= prof) next
    y_max_zona <- min(rng[2], prof)
    y_min_zona <- rng[1]
    pct        <- zonas_pct[i]
    if (is.na(pct) || pct < 2) next
    
    dat_zona <- ang_data %>% filter(zona_num == i)
    if (nrow(dat_zona) == 0) next
    
    angulo_mean <- dat_zona$angulo_mean[1]
    long_1      <- (pct / 100) * 2.5
    long_2      <- long_1 * 0.25
    
    n_vis_1 <- max(1, round(dat_zona$n_laterales[1] / 60))
    n_vis_1 <- min(n_vis_1, 3)
    
    y_lats <- seq(y_min_zona + 1.5, y_max_zona - 1.5, length.out = n_vis_1)
    y_lats <- y_lats[y_lats <= prof]
    
    set.seed(i * 100 + which(genotipos_presentes == gen_nombre))
    
    for (j in seq_along(y_lats)) {
      y_l     <- y_lats[j]
      idx_piv <- which.min(abs(y_pts - y_l))
      x_orig  <- x_pts[idx_piv]
      
      angulo_usar <- angulo_mean + runif(1, -8, 8)
      angulo_usar <- max(10, min(85, angulo_usar))
      
      for (lado in c(-1, 1)) {
        ang_rad_1 <- angulo_usar * pi / 180
        dx1 <- long_1 * sin(ang_rad_1) * lado
        dy1 <- long_1 * cos(ang_rad_1) * ratio
        
        x1_end <- x_orig + dx1
        y1_end <- y_l    + dy1
        
        lats <- rbind(lats, data.frame(
          x0 = x_orig, y0 = y_l,
          x1 = x1_end, y1 = y1_end,
          orden = "1"
        ))
        
        ang_rad_2 <- angulo_usar * pi / 180
        dx2 <- long_2 * sin(ang_rad_2) * lado
        dy2 <- long_2 * cos(ang_rad_2) * ratio
        
        x_mid <- x_orig + dx1 * 0.5
        y_mid <- y_l    + dy1 * 0.5
        
        lats <- rbind(lats, data.frame(
          x0 = x_mid, y0 = y_mid,
          x1 = x_mid + dx2, y1 = y_mid + dy2,
          orden = "2"
        ))
      }
    }
  }
  
  color_gen <- colores[gen_nombre]
  y_tort    <- min(40, prof * 0.60)
  idx_tort  <- which.min(abs(y_pts - y_tort))
  x_tort    <- x_pts[idx_tort] + 0.3
  
  # Lineas y etiquetas de zona: SIEMPRE sobre el rango global, no sobre
  # la profundidad individual de este genotipo. Asi todos los esquemas
  # comparten exactamente las mismas divisiones y son comparables entre si.
  y_breaks_global <- seq(alto_zona, PROF_TOTAL_RIZOTRON, by = alto_zona)
  y_breaks_global <- y_breaks_global[y_breaks_global < PROF_TOTAL_RIZOTRON]
  y_mid_global    <- seq(alto_zona/2, by = alto_zona,
                         length.out = length(zonas_lim))
  y_inicio_global <- seq(0, by = alto_zona, length.out = length(zonas_lim))
  y_fin_global    <- seq(alto_zona, by = alto_zona, length.out = length(zonas_lim))
  
  p <- ggplot() +
    geom_hline(yintercept = y_breaks_global,
               color = "gray80", linewidth = 0.2, linetype = "dashed") +
    
    { if (nrow(lats) > 0)
      geom_segment(
        data = lats %>% filter(orden == "2"),
        aes(x = x0, y = y0, xend = x1, yend = y1),
        color = color_gen, linewidth = 0.5, alpha = 0.65
      )
      else NULL
    } +
    
    { if (nrow(lats) > 0)
      geom_segment(
        data = lats %>% filter(orden == "1"),
        aes(x = x0, y = y0, xend = x1, yend = y1),
        color = color_gen, linewidth = 0.75, alpha = 0.9
      )
      else NULL
    } +
    
    geom_path(data = piv, aes(x = x, y = y),
              color = color_gen, linewidth = 1.0) +
    
    annotate("text", x = x_tort, y = y_tort,
             label = sprintf("T=%.2f", tort),
             size = 2.5, color = color_gen,
             hjust = 0, fontface = "italic") +
    
    { if (mostrar_eje)
      annotate("text", x = -2.0, y = y_mid_global,
               label = sprintf("%.0f-%.0f cm", y_inicio_global, y_fin_global),
               size = 2.0, color = "gray55", hjust = 1)
      else NULL
    } +
    
    { if (mostrar_eje_der)
      annotate("text", x = 2.3, y = y_mid_global,
               label = sprintf("%.0f-%.0f cm", y_inicio_global, y_fin_global),
               size = 2.0, color = "gray55", hjust = 0)
      else NULL
    } +
    
    scale_y_reverse(limits = c(PROF_TOTAL_RIZOTRON + 3, -5), expand = c(0, 0)) +
    scale_x_continuous(limits = c(-4.5, 4.5), expand = c(0, 0)) +
    coord_fixed(ratio = 1/ratio) +
    
    labs(title = gen_nombre, x = NULL, y = NULL) +
    
    theme_minimal(base_size = 10) +
    theme(
      legend.position = "none",
      panel.grid      = element_blank(),
      axis.text       = element_blank(),
      axis.ticks      = element_blank(),
      plot.title      = element_text(face = "bold", size = 11,
                                     color = color_gen, hjust = 0.5),
      plot.margin     = margin(5, 2, 5, 2)
    )
  
  return(p)
}

df_arq <- df_gen %>%
  select(genotipo, prof_final_mean, tort_mean,
         dens_ram_mean, n_lat_mean) %>%
  left_join(df_zonas %>% select(genotipo, all_of(zonas_cols_pct)),
            by = "genotipo") %>%
  mutate(genotipo = factor(genotipo,
                           levels = df_gen %>%
                             arrange(desc(prof_final_mean)) %>%
                             pull(genotipo)))

gen_orden <- df_gen %>% arrange(desc(prof_final_mean)) %>% pull(genotipo)

# Panel combinado con layout definido por el usuario
n_col_esquemas <- ifelse(is.null(N_COL_ESQUEMAS),
                         ceiling(length(gen_orden) / N_ROW_ESQUEMAS),
                         N_COL_ESQUEMAS)

# Generar y guardar cada esquema individual + acumular para el panel
plots <- purrr::imap(gen_orden, function(gen, idx) {
  # Posicion de este genotipo dentro de la grilla
  fila_actual     <- ceiling(idx / n_col_esquemas)
  col_actual      <- idx - (fila_actual - 1) * n_col_esquemas
  es_primero_fila <- (col_actual == 1)
  es_ultimo_fila  <- (col_actual == n_col_esquemas) |
    (idx == length(gen_orden))
  
  # Version para el panel grupal: eje izq. solo en col 1, eje der. solo en ultima col
  p <- gen_esquema_panel(
    gen,
    df_arq       %>% filter(genotipo == gen),
    df_ang_rango %>% filter(genotipo == gen),
    zonas_cols      = zonas_cols_pct,
    mostrar_eje     = es_primero_fila,
    mostrar_eje_der = es_ultimo_fila
  )
  
  # Version individual: siempre con ambos ejes y nombre del genotipo
  p_individual <- gen_esquema_panel(
    gen,
    df_arq       %>% filter(genotipo == gen),
    df_ang_rango %>% filter(genotipo == gen),
    zonas_cols      = zonas_cols_pct,
    mostrar_eje     = TRUE,
    mostrar_eje_der = TRUE
  )
  
  ggsave(file.path(CARPETA_SALIDA, "esquemas_individuales",
                   sprintf("esquema_%s.png", gen)),
         p_individual, width = 4.5, height = 6, dpi = 300)
  
  return(p)
})

cat(sprintf("Guardados %d esquemas individuales en: esquemas_individuales/\n",
            length(plots)))

panel_esquemas <- wrap_plots(plots, nrow = N_ROW_ESQUEMAS, ncol = n_col_esquemas) +
  plot_annotation(
    title    = "Root system architecture schemes by genotype",
    subtitle = "Lateral length ∝ zone coverage | Angles from measured data | Taproot path is a simulated representation calibrated to measured tortuosity (not the actual root trajectory)",
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(color = "gray40", size = 8)
    )
  )

panel_esquemas
ggsave(file.path(CARPETA_SALIDA, "esquemas_panel.png"),
       panel_esquemas,
       width  = 3.2 * n_col_esquemas,
       height = 6   * N_ROW_ESQUEMAS,
       dpi = 300)
cat("Guardada: esquemas_panel.png\n")
# Para exportar con otro tamano:
# ggsave(file.path(CARPETA_SALIDA, "esquemas_panel.png"),
#        panel_esquemas, width = 16, height = 10, dpi = 300)

cat(sprintf("\nTodos los resultados guardados en: %s\n", CARPETA_SALIDA))
