# Install core R packages used by this Quarto workflow.
required <- c("knitr", "rmarkdown", "ggplot2", "quarto", "reticulate")
missing <- required[!(required %in% installed.packages()[, "Package"])]

if (length(missing) > 0) {
  install.packages(missing)
}

cat("R package setup complete.\n")
