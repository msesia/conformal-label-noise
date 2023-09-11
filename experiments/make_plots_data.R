options(width = 300)

library(tidyverse)
library(latex2exp)
library(RColorBrewer)

load_data <- function(exp.num) {
    idir <- sprintf("results_hpc/exp%d", exp.num)
    ifile.list <- list.files(idir)
    results <- do.call("rbind", lapply(ifile.list, function(ifile) {
        df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols(), guess_max=2)
    }))


    summary <- results %>%
        pivot_longer(c("Coverage", "Size"), names_to = "Key", values_to = "Value") %>%
        group_by(data, K, n_cal, n_test, epsilon_n_clean, epsilon_n_corr, estimate, Guarantee, Alpha, Label, Method, Key) %>%
        summarise(Mean=mean(Value), N=n(), SE=2*sd(Value)/sqrt(N))

    return(summary)
}

exp.num <- 101
summary <- load_data(exp.num)

if(FALSE) {
    method.values = c("Standard", "Adaptive (pessimistic)", "Adaptive (optimistic)")
    method.labels = c("Standard", "Adaptive", "Adaptive+")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    df.dummy <- tibble(key="Coverage", value=0.95)
    df.dummy2 <- tibble(key="Coverage", value=0.5)
    color.scale <- cbPalette[c(1,3,7)]
    shape.scale <- c(1,2,0)
    linetype.scale <- c(1,1,1)
} else {
    method.values = c("Standard-none", "Adaptive (optimistic)-none", "Adaptive (optimistic)-rho-epsilon-point")
    method.labels = c("Standard", "Adaptive+", "Adaptive+ (plug-in)")
    cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    df.dummy <- tibble(key="Coverage", value=0.95)
    df.dummy2 <- tibble(key="Coverage", value=0.5)
    color.scale <- cbPalette[c(1,7,7)]
    shape.scale <- c(1,0,15)
    linetype.scale <- c(1,1,1)
}

plot.alpha <- 0.1
plot.guarantee <- "lab-cond"
plot.K <- 4
plot.epsilon <- 0.1


# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the calibration set size, for different values of contamination parameter

make_figure_1 <- function(plot.alpha=0.1, plot.guarantee="marginal", save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(101)
    }

    df <- summary %>%
        filter(Alpha==plot.alpha, Guarantee==plot.guarantee, Label=="marginal")
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.88,0.94), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = sprintf("%s-%s", Method, estimate)) %>%
        filter(Method %in% c(method.values)) %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        facet_wrap(.~Key, scales="free") +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
        scale_x_continuous(trans='log10', limits=c(500,10000)) +
        xlab("Number of calibration samples") +
        ylab("") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


    if(save_plots) {
        plot.file <- sprintf("figures/cifar10_%s.pdf", plot.guarantee)
        ggsave(file=plot.file, height=2.5, width=6, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data(exp.num)
make_figure_1(plot.alpha=0.1, plot.guarantee="marginal", save_plots=TRUE, reload=FALSE)
make_figure_1(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)
