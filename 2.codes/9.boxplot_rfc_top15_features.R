# setwd('C:/Users/Mirac/Desktop/data_reanalysis_20211221/1.data_generated')

library(ggplot2)
library(readxl)
library(forcats)
library(ggpubr)
library(pacman)
library(ggsignif)

gold_ratiso_all_samples <-read_excel("8.2.oav_only_boxplot_dataset4ggplot_2021-12-22.xlsx")


gold_ratiso_all_samples$compounds <- fct_inorder(gold_ratiso_all_samples$compounds)

p <- ggplot(gold_ratiso_all_samples, aes(compounds, ratio, fill=factor(level)))

p <- p + geom_boxplot(outlier.shape = NA, na.rm = TRUE) + 
  geom_point(position = position_jitterdodge(), size=0.5) +
  ylab('Log2Ratios') + xlab('Comparing compounds') + 
  scale_y_continuous() +
  scale_fill_manual(values = c('#FF9966', '#FFFFCC'))


p  + theme_bw() + theme(axis.title.x=element_blank(),
                        axis.text.x=element_blank(),
                        axis.ticks.x=element_blank()) +
  facet_wrap(~compounds,scales="free", nrow=3)


ggsave(filename='../3.figures/8.3.top15features_boxplot.pdf', 
       width = 15, height = 7, 
       device = cairo_pdf, family = "sans")