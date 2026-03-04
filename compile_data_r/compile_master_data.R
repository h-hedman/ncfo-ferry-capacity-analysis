################################################################################
# NCFO MASTER DATASET BUILDER
# Creates unified datasets for:
#   operator_segment
#   segment
#   terminal
#   vessel
################################################################################

library(readr)
library(readxl)
library(dplyr)
library(janitor)

################################################################################
# PATHS
################################################################################

raw_path <- "G:/My Drive/Nexus/Employment/9_U.S. Department of Transportation (DOT) - ORISE/DOT - ORISE/Projects/Survey Data/data"

out_path <- "G:/My Drive/Nexus/Employment/9_U.S. Department of Transportation (DOT) - ORISE/DOT - ORISE/Projects/NCFO ML/data/processed"

################################################################################
# UNIVERSAL LOADER
################################################################################

load_survey <- function(path) {
  
  ext <- tools::file_ext(path)
  
  df <- switch(ext,
               
               "csv"  = read_csv(path,
                                 col_types = cols(.default = "c")),
               
               "xlsx" = read_excel(path,
                                   col_types = "text"),
               
               stop("Unsupported file format")
  )
  
  colnames(df) <- toupper(colnames(df))
  
  return(df)
}

################################################################################
# SEGMENT DATASET
################################################################################

seg_2018 <- load_survey(file.path(raw_path,
                                  "bts_osp_national_census_ferry_operators_2018_Segment_2020_04_23.xlsx"))

seg_2019 <- load_survey(file.path(raw_path,
                                  "2020_NCFO_Segments_File.csv"))

seg_2022 <- load_survey(file.path(raw_path,
                                  "2022_NCFO_Segments_File.csv"))

seg_2018$DATA_YEAR <- 2018
seg_2019$DATA_YEAR <- 2019
seg_2022$DATA_YEAR <- 2022

segment_master <- bind_rows(
  seg_2022,
  seg_2019,
  seg_2018
)

write_csv(segment_master,
          file.path(out_path,"segment_master.csv"))

################################################################################
# VESSEL DATASET
################################################################################

ves_2018 <- load_survey(file.path(raw_path,
                                  "bts_osp_national_census_ferry_operators_2018_Vessel_2019_11_27.xlsx"))

ves_2019 <- load_survey(file.path(raw_path,
                                  "2020_NCFO_Vessels_File.csv"))

ves_2022 <- load_survey(file.path(raw_path,
                                  "2022_NCFO_Vessels_File.csv"))

ves_2018$DATA_YEAR <- 2018
ves_2019$DATA_YEAR <- 2019
ves_2022$DATA_YEAR <- 2022

vessel_master <- bind_rows(
  ves_2022,
  ves_2019,
  ves_2018
)

write_csv(vessel_master,
          file.path(out_path,"vessel_master.csv"))

################################################################################
# TERMINAL DATASET
################################################################################

term_2018 <- load_survey(file.path(raw_path,
                                   "bts_osp_national_census_ferry_operators_2018_Terminal_2020-04-01.xlsx"))

term_2019 <- load_survey(file.path(raw_path,
                                   "2020_NCFO_Terminals_File.csv"))

term_2022 <- load_survey(file.path(raw_path,
                                   "2022_NCFO_Terminals_File.csv"))

term_2018$DATA_YEAR <- 2018
term_2019$DATA_YEAR <- 2019
term_2022$DATA_YEAR <- 2022

terminal_master <- bind_rows(
  term_2022,
  term_2019,
  term_2018
)

write_csv(terminal_master,
          file.path(out_path,"terminal_master.csv"))

################################################################################
# OPERATOR SEGMENT DATASET
################################################################################

opseg_2018 <- load_survey(file.path(raw_path,
                                    "bts_osp_national_census_ferry_operators_2018_OperatorSegment_2021_2_12.xlsx"))

opseg_2019 <- load_survey(file.path(raw_path,
                                    "2020_NCFO_Operator_Segment_File.csv"))

opseg_2022 <- load_survey(file.path(raw_path,
                                    "2022_NCFO_Operator_Segments_File.csv"))

opseg_2018$DATA_YEAR <- 2018
opseg_2019$DATA_YEAR <- 2019
opseg_2022$DATA_YEAR <- 2022

operator_segment_master <- bind_rows(
  opseg_2022,
  opseg_2019,
  opseg_2018
)

write_csv(operator_segment_master,
          file.path(out_path,"operator_segment_master.csv"))

################################################################################
# RUN CHECK
################################################################################

cat("\nNCFO Master datasets created:\n")
cat("segment_master.csv\n")
cat("vessel_master.csv\n")
cat("terminal_master.csv\n")
cat("operator_segment_master.csv\n\n")