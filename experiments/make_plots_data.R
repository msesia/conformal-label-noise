options(width = 300)
summary
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
        ggsave(file=plot.file, height=2.25, width=6.5, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data(exp.num)
make_figure_1(plot.alpha=0.1, plot.guarantee="marginal", save_plots=TRUE, reload=FALSE)
make_figure_1(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)



make_figure_1_slides <- function(plot.alpha=0.1, plot.guarantee="marginal", save_plots=FALSE, reload=FALSE, version=1) {
    if(reload) {
        summary <- load_data(101)
    }

    df <- summary %>%
        filter(Alpha==plot.alpha, Guarantee==plot.guarantee, Label=="marginal")
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage", "Size", "Size"), Mean=c(0.88,0.94, 1.1, 1.3), n_cal=1000, Method="Standard")

    df <- df %>%
        mutate(Method = sprintf("%s-%s", Method, estimate)) %>%
        filter(Method %in% c(method.values))

    if(version==2) {
        df <- df %>% filter(Method %in% c("Standard-none", "Adaptive (optimistic)-none"))
    } else if (version==1) {
        df <- df %>% filter(Method=="Standard-none")
    }
    
    pp <- df %>%
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
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.position="bottom")


    if(save_plots) {
        plot.file <- sprintf("figures_slides/cifar10_%s_%s.pdf", plot.guarantee, version)
        ggsave(file=plot.file, height=3.5, width=6, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

make_figure_1_slides(plot.alpha=0.1, plot.guarantee="marginal", save_plots=TRUE, reload=FALSE, version=1)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="marginal", save_plots=TRUE, reload=FALSE, version=2)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="marginal", save_plots=TRUE, reload=FALSE, version=3)

make_figure_1_slides(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE, version=1)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE, version=2)
make_figure_1_slides(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE, version=3)



# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the calibration set size, for different values of contamination parameter
# Include the naive theoretical benchmark

method.values = c("Standard-none", "Standard (theory)-none", "Adaptive (optimistic)-none")
method.labels = c("Standard", "Standard (WC multiplicative)", "Adaptive+")
cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
df.dummy <- tibble(key="Coverage", value=0.95)
df.dummy2 <- tibble(key="Coverage", value=0.5)
color.scale <- cbPalette[c(1,1,7,7)]
shape.scale <- c(1,8,0,15)
linetype.scale <- c(1,3,1,1)


make_figure_2 <- function(plot.alpha=0.1, plot.guarantee="marginal", save_plots=FALSE, reload=FALSE) {
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
        plot.file <- sprintf("figures/cifar10_%s_theory.pdf", plot.guarantee)
        ggsave(file=plot.file, height=2.25, width=7, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data(exp.num)
make_figure_2(plot.alpha=0.1, plot.guarantee="marginal", save_plots=TRUE, reload=FALSE)
make_figure_2(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)


# --------------------------------------------------
# Plot marginal coverage (for marginal and label-conditional calibration) as a function of the calibration set size, for different values of contamination parameter
# Include the theoretical bands for the coverage

method.values = c("Standard-none", "Lower", "Upper")
method.labels = c("Standard", "Lower", "Upper")
cbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
df.dummy <- tibble(key="Coverage", value=0.95)
df.dummy2 <- tibble(key="Coverage", value=0.5)
color.scale <- cbPalette[c(1,1,7,7)]
shape.scale <- c(1,0,15)
linetype.scale <- c(1,1,1)

make_figure_2_bands <- function(plot.alpha=0.1, plot.guarantee="marginal", save_plots=FALSE, reload=FALSE) {
    if(reload) {
        summary <- load_data(101)
    }

    df.theory.1 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha-Epsilon*(1-1/K),
                          Method="Lower", Theory="WC bounds (additive)", Key="Coverage")
    if(plot.guarantee=="marginal") {
        df.theory.2 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha+1/(n_cal+1)+Epsilon*(1-1/K),
                              Method="Upper", Theory="WC bounds (additive)", Key="Coverage")
    df.theory.4 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha+1/(n_cal+1)+Epsilon*(1-1/K),
                          Method="Upper", Theory="WC bounds (multiplicative)", Key="Coverage")
    } else {
        df.theory.2 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha+1/(n_cal/K+1)+Epsilon*(1-1/K),
                              Method="Upper", Theory="WC bounds (additive)", Key="Coverage")
        df.theory.4 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha+1/(n_cal/K+1)+Epsilon*(1-1/K),
                              Method="Upper", Theory="WC bounds (multiplicative)", Key="Coverage")
    }
    df.theory.3 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha/(1-Epsilon*(1-1/K)),
                          Method="Lower", Theory="WC bounds (multiplicative)", Key="Coverage")

    df.theory.5 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha-(Epsilon/(1-Epsilon))*(1-1/K),
                          Method="Lower", Theory="WC bounds (Corollary S1)", Key="Coverage")
    if(plot.guarantee=="marginal") {
        df.theory.6 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha+1/(n_cal+1)+(Epsilon/(1-Epsilon))*(1-1/K),
                              Method="Upper", Theory="WC bounds (Corollary S1)", Key="Coverage")
    } else {
        df.theory.6 <- tibble(n_cal=unique(summary$n_cal), Alpha=0.1, K=10, Epsilon=0.051, Mean=1-Alpha+1/(n_cal/K+1)+Epsilon*(1-1/K),
                              Method="Upper", Theory="WC bounds (Corollary S1)", Key="Coverage")
    }

    df.theory <- rbind(df.theory.1, df.theory.2, df.theory.3, df.theory.4, df.theory.5, df.theory.6) %>%
        mutate(Method = factor(Method, method.values, method.labels))        
    
    df <- summary %>%
        filter(Alpha==plot.alpha, Guarantee==plot.guarantee, Label=="marginal")
    df.nominal <- tibble(Key="Coverage", Mean=1-plot.alpha)
    df.range <- tibble(Key=c("Coverage","Coverage"), Mean=c(0.88,0.94), n_cal=1000, Method="Standard")
    pp <- df %>%
        mutate(Method = sprintf("%s-%s", Method, estimate)) %>%
        filter(Method %in% c(method.values), Key=="Coverage") %>%
        mutate(Method = factor(Method, method.values, method.labels)) %>%
        ggplot(aes(x=n_cal, y=Mean, color=Method, shape=Method, linetype=Method)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE), width=0.1) +
        geom_line(data=df.theory, color="black", size=0.75) +
        facet_grid(.~Theory) +
        geom_hline(data=df.nominal, aes(yintercept=Mean), linetype="dashed") +
        geom_point(data=df.range, aes(x=n_cal, y=Mean), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        scale_linetype_manual(values=linetype.scale) +
        scale_x_continuous(trans='log10', limits=c(500,10000)) +
        xlab("Number of calibration samples") +
        ylab("Coverage") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), legend.position="none")


    if(save_plots) {
        plot.file <- sprintf("figures/cifar10_%s_theory_bands.pdf", plot.guarantee)
        ggsave(file=plot.file, height=2.25, width=6.5, units="in")
        return(NULL)
    } else{
        return(pp)
    }
}

summary <- load_data(exp.num)
make_figure_2_bands(plot.alpha=0.1, plot.guarantee="marginal", save_plots=TRUE, reload=FALSE)
make_figure_2_bands(plot.alpha=0.1, plot.guarantee="lab-cond", save_plots=TRUE, reload=FALSE)
