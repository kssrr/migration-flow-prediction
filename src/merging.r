library(dplyr)
library(tidyr)
library(purrr)
library(stringr)

# Import migration data:
mig <- readr::read_csv(here::here("data/raw/world_bank_migration_raw.csv"))

# Import all other data files:
files <- list.files(here::here("data"), pattern = "\\.csv$")

files |> 
  map(\(f) readr::read_csv(here::here("data", f))) |> 
  setNames(str_remove(files, ".csv")) |> 
  list2env(rlang::global_env())

# 1 - cleaning the migration data from the world bank ----

# There are a number of composite records (e.g. "Europe", "Subsaharan Africa"
# etc); making sure we only retain *actual* countries:

actual_countries <- 
  countrycode::codelist |> 
  select(name = country.name.en, iso3 = iso3c) |> 
  drop_na(iso3) |> 
  pull(iso3)

mig <- 
  mig |> 
  select(iso3 = `Country Code`, `1960`:`2023`) |> 
  pivot_longer(cols = -iso3, names_to = "year", values_to = "net_migration") |> 
  filter(iso3 %in% actual_countries) |> 
  mutate(year = as.numeric(year))

# 2 - Joining in features ----

features <- list(
  conflict_deaths, gdp, gdp_growth, internet_usage, liberal_democracy,
  natural_disasters, pop_growth, population, sanctions, unemployment_youth
)

full <- 
  features |> 
  reduce(\(x, y) left_join(x, y, by = c("iso3", "year")), .init = mig) |> 
  mutate(across(conflict_deaths:unemployment_youth, as.numeric))

# 3 - Export ----

full |> readr::write_csv(here::here("data/final/full.csv"))
