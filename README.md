# Rain Watch: Short-Term Rainfall Nowcasting with ConvLSTM

Deep Learning (DSE3120) course project.

Predicts rainfall 30 and 60 minutes ahead from the last 3 hours of satellite rain maps over the Maharashtra coast (16-22°N, 70-76°E), and turns the forecasts into city alerts in a Streamlit dashboard.

## What it does
- **Data:** NASA GPM IMERG half-hourly rainfall (Version 07), July 2024, 31 days, cropped to 60 x 60 pixels (about 10 km each).
- **Windows:** 6 past maps in, 2 future maps out. Split by time (70/15/15) with a gap at each boundary, giving 1,036 train, 214 validation and 215 test windows.
- **Models:** persistence baseline, CNN, ConvLSTM, and a ConvLSTM trained with extra weight on heavy-rain pixels (5 mm/hr or more count 6 times more).
- **Dashboard:** alert cards for six cities, rain map, six-hour storm replay, city timeline and a reliability tab.

## Results (215 test forecasts, heavy rain = 5 mm/hr or more)

| Model | Lead (min) | MSE | Recall | CSI |
|---|---|---|---|---|
| Persistence | 30 | 0.639 | 0.607 | 0.434 |
| CNN | 30 | 0.440 | 0.520 | 0.454 |
| ConvLSTM | 30 | 0.449 | 0.511 | 0.449 |
| ConvLSTM + heavy-rain weight | 30 | 0.497 | 0.658 | 0.464 |
| Persistence | 60 | 1.005 | 0.469 | 0.304 |
| CNN | 60 | 0.679 | 0.335 | 0.294 |
| ConvLSTM | 60 | 0.677 | 0.350 | 0.304 |
| ConvLSTM + heavy-rain weight | 60 | 0.748 | 0.530 | 0.352 |

Weighting the loss catches more heavy rain at the cost of about 10% higher MSE and more false alarms. City level (six cities, 30 min, 5 mm/hr): 69% of rain cases warned, 23% of warnings wrong, CSI 0.57.

## Project structure
```
src/
  download_range.py   download and crop IMERG one day at a time
  build_dataset.py    crop helper and quick statistics
  make_windows.py     build train/validation/test windows
  metrics.py          MSE, CSI, POD
  baseline.py         persistence baseline
  models.py           CNN and ConvLSTM definitions
  train.py            training script (run in Colab)
  dashboard.py        Rain Watch Streamlit dashboard
results/              saved predictions used by the dashboard
reports/              figures
```

## How to run
1. Install: `pip install -r requirements.txt`, plus `pip install streamlit plotly pandas` for the dashboard.
2. Download data (needs a free NASA Earthdata account): `python src/download_range.py`
3. Build windows: `python src/make_windows.py`
4. Train in Google Colab (GPU): `python src/train.py --model convlstm --heavy_weight 5`
5. Dashboard: `streamlit run src/dashboard.py`

## Limitations
- One month of monsoon rain and a test period of about 4.5 days, so scores are indicative.
- Forecasts are smoother than reality: peak intensity is underestimated.
- The dashboard replays saved predictions and is not connected to live data. This is a student prototype, not an official forecast.
