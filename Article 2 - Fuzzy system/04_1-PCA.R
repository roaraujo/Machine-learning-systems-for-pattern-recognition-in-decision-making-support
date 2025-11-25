
# Carregar as bibliotecas necessárias
library(ggbiplot)
library(readxl)
library(dplyr)

data <- read_excel("C:\\Users\\User\\Documents\\Mestrado - UFBA\\Dissertação\\mestrado-ufba\\Artigo 1\\X_train_pca.xlsx")

data2 <- data %>% select(-GR)
data2
data_std <- scale(data2)
# Realizar PCA
pc <- prcomp(data_std, center = TRUE, scale. = TRUE)

pc
g <- ggbiplot(pc,
              obs.scale = 1,
              var.scale = 1,
              groups =data$GR,
              ellipse = TRUE,
              circle = TRUE,
              ellipse.prob = 0.9)
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal',
               legend.position = 'top')

print(g)


# Gerar gráfico com ggplot
g <- ggbiplot(pc,
              obs.scale = 1,
              var.scale = 1,
              ellipse = TRUE,
              circle = TRUE,
              ellipse.prob = 0.68)

# Exibir o gráfico
print(g)
