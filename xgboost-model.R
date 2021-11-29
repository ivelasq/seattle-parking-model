library(tidyverse)
library(RSocrata)
library(lubridate)
library(usemodels)
library(tidymodels)

# Site: https://data.seattle.gov/Transportation/Paid-Parking-Occupancy-Last-30-Days-/rke9-rsvs

# Data Cleaning

parking_data <-
  RSocrata::read.socrata(
    "https://data.seattle.gov/resource/rke9-rsvs.json?$where=sourceelementkey <= 1050"
  )

parking_id <-
  parking_data %>% 
  group_by(blockfacename, location.coordinates) %>% 
  mutate(id = cur_group_id()) %>% 
  ungroup()

parking_clean <-
  parking_id %>%
  mutate(across(c(parkingspacecount, paidoccupancy), as.numeric),
         occupancy_pct = paidoccupancy/parkingspacecount) %>%
  group_by(
    id = id,
    hour = as.numeric(hour(occupancydatetime)),
    month = as.numeric(month(occupancydatetime)),
    dow = as.numeric(wday(occupancydatetime)),
    date = date(occupancydatetime)
  ) %>%
  summarize(occupancy_pct = mean(occupancy_pct, na.rm = TRUE)) %>%
  drop_na() %>% 
  ungroup()

parking_information <-
  parking_id %>% 
  mutate(loc = location.coordinates) %>% 
  select(id, blockfacename, loc) %>% 
  distinct(id, blockfacename, loc) %>% 
  unnest_wider(loc) %>% 
  rename(lon = `...1`,
         lat = `...2`)

parking_split <-
  parking_clean %>%
  arrange(date) %>%
  select(-date) %>% 
  initial_time_split(prop = 0.75)

# Model

xgboost_recipe <-
  recipe(formula = occupancy_pct ~ ., data = parking_clean) %>%
  step_zv(all_predictors())  %>%
  prep()

xgboost_folds <-
  recipes::bake(xgboost_recipe,
                new_data = training(parking_split)) %>%
  rsample::vfold_cv(v = 5)

xgboost_model <-
  boost_tree(
    mode = "regression",
    trees = 1000,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) %>%
  set_engine("xgboost", objective = "reg:squarederror")
  
xgboost_params <-
  parameters(min_n(),
             tree_depth(),
             learn_rate(),
             loss_reduction())

xgboost_grid <-
  grid_max_entropy(xgboost_params,
                   size = 5)

xgboost_wf <-
  workflows::workflow() %>%
  add_model(xgboost_model) %>%
  add_formula(occupancy_pct ~ .)

xgboost_tuned <- tune::tune_grid(
  object = xgboost_wf,
  resamples = xgboost_folds,
  grid = xgboost_grid,
  metrics = yardstick::metric_set(rmse, rsq, mae),
  control = tune::control_grid(verbose = TRUE)
)

xgboost_best <-
  xgboost_tuned %>%
  tune::select_best("rmse")

xgboost_final <-
  xgboost_model %>%
  finalize_model(xgboost_best)

# Create Objects

train_processed <-
  bake(xgboost_recipe, new_data = training(parking_split))

prediction_fit <-
  xgboost_final %>%
  fit(formula = occupancy_pct ~ .,
      data    = train_processed)

model_details <- list(model = xgboost_final,
                      recipe = xgboost_recipe,
                      prediction_fit = prediction_fit)

rsc <-
  pins::board_rsconnect(server = Sys.getenv("CONNECT_SERVER"),
                        key = Sys.getenv("CONNECT_API_KEY"))

pins::pin_write(
  board = rsc,
  x = model_details,
  name = "seattle_parking_model",
  description = "Seattle Occupancy Percentage XGBoost Model",
  type = "rds"
)

pins::pin_write(
  board = rsc,
  x = parking_information,
  name = "seattle_parking_info",
  description = "Seattle Parking Information",
  type = "rds"
)

# Evaluation

train_processed <-
  bake(xgboost_recipe, new_data = training(parking_split))

test_processed  <-
  bake(xgboost_recipe, new_data = testing(parking_split))

test_prediction <-
  xgboost_final %>%
  fit(formula = occupancy_pct ~ .,
      data    = train_processed) %>%
  predict(new_data = test_processed) %>%
  bind_cols(testing(parking_split))

xgboost_score <-
  test_prediction %>%
  yardstick::metrics(occupancy_pct, .pred) %>%
  mutate(.estimate = format(round(.estimate, 3), big.mark = ",")) %>%
  knitr::kable()

write_csv(parking_information, "parking_information.csv")
