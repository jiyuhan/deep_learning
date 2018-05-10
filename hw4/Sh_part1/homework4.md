Homework 4 
---
### Part 1
- Implemented `generate_x_y_data_v5`, `should_buy(x0, x1, x2, x3, x4, x5)`, 
`buy(cash, shares, price)`, and `sell(cash, shares, price)`.
- The stock simulation is done in `stock_simulation_part1(isTest)` 
with $10,000 and 0 shares using hte model built over 300 days.

| Hidden_dim | Iterations | Batch Size | Final Cash | Final Shares |
| :--------: |:----------:| :---------:| :-----------:| :---------:|
| 200        |    500     |   100      | $113064.33| 3 |
| 500        | 500        |   100      | $12 |
| 1000       | 500        |   100      | $1 |
| 200        |    1000    |   250      | $1600 | 
| 500        | 1000       |   250      | $12 |
| 1000       | 1000       |    250     | error | error |