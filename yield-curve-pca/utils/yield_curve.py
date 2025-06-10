import QuantLib as ql


def construct_yc(today, calendar, settlement, face_amount, frequency,
                 day_count, busi_conv, end_of_month, row):  
    # Extract par yields and deposit rates from row
    depo_rates = {}
    par_yields = {}
    for ind, v in row.items():
        if not v: continue
        if ind < 1:
            depo_rates[int(ind * 12)] = v
        else:
            par_yields[int(ind * 12)] = v

    # Construct deposit/bond helpers
    helpers = []
    for k, v in depo_rates.items():
        maturity = ql.Period(k, ql.Months)
        helper = ql.DepositRateHelper( # Might have bug here bc conventions
            ql.QuoteHandle(ql.SimpleQuote(v/100.0)),
            maturity,
            settlement,
            calendar,
            busi_conv,
            end_of_month,
            day_count)
            
        helpers.append(helper)
    
    for k, v in par_yields.items():
        maturity = ql.Period(k, ql.Months)
        schedule = ql.Schedule(
            today,
            today + maturity,
            ql.Period(frequency),
            calendar,
            busi_conv,
            busi_conv,
            ql.DateGeneration.Backward,
            False
        )
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(face_amount)),
            settlement,
            face_amount,
            schedule,
            [v/100],
            day_count
        )
        helpers.append(helper)

    # Construct spline-fitted yield curve
    return ql.PiecewiseLogCubicDiscount(
        today,
        helpers,
        day_count
    )

def bootstrap_interpolate(row, new_tenors, plot=False):
    # Params
    date = row.name
    today = ql.Date(date.day, date.month, date.year)
    ql.Settings.instance().evaluationDate = today

    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    settlement = 0
    face_amount = 100
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISMA)
    busi_conv = ql.Unadjusted
    end_of_month = True

    # Construct yield curve
    curve = construct_yc(today, calendar, settlement, face_amount,
                         frequency, day_count, busi_conv, end_of_month, 
                         row)
    curve_handle = ql.YieldTermStructureHandle(curve)

    # Solve for interpolated par rates
    def get_par_rates(months): # Only works for [6m+)
        maturity_date = today + ql.Period(months, ql.Months)
        schedule = ql.Schedule(
            today,
            maturity_date,
            ql.Period(ql.Semiannual),
            calendar,
            busi_conv, 
            busi_conv,
            ql.DateGeneration.Backward, False
        )
    
        def objective(coupon):
            bond = ql.FixedRateBond(
                0,
                face_amount,
                schedule,
                [coupon/100],
                day_count
            )
            bond.setPricingEngine(ql.DiscountingBondEngine(curve_handle))
            return bond.NPV() - 100
    
        solver = ql.Brent()
        return solver.solve(objective, 1e-8, 3, 0.0001, 15)

    par_rates = [row.iloc[0]]
    for y in new_tenors[1:]:
        par_rates.append(get_par_rates(int(y*12)))

    return par_rates


def price_bond(yc_row, tenor, rate):
    # Params
    date = yc_row.name
    today = ql.Date(date.day, date.month, date.year)
    ql.Settings.instance().evaluationDate = today
    
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    settlement = 0
    face_amount = 100
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISMA)
    busi_conv = ql.Unadjusted
    end_of_month = True
    
    # Extract par yields and deposit rates from row
    depo_rates = {}
    par_yields = {}
    
    for ind, v in yc_row.items():
        if not v: continue
        if ind < 1:
            depo_rates[int(ind * 12)] = v
        else:
            par_yields[int(ind * 12)] = v
    
    # Construct yield curve
    curve = construct_yc(today, calendar, settlement, face_amount,
                         frequency, day_count, busi_conv, end_of_month, 
                         yc_row)
    curve_handle = ql.YieldTermStructureHandle(curve)

    # Price new bond
    issue_date = today
    maturity_date = today + ql.Period(tenor, ql.Years)
    schedule = ql.Schedule(
        issue_date,
        maturity_date,
        ql.Period(ql.Semiannual),
        calendar,
        busi_conv,
        busi_conv,
        ql.DateGeneration.Backward,
        False
    )
    
    bond = ql.FixedRateBond(
        settlement,  # settlement days (0 if curve is already today)
        100,  # face value
        schedule,
        [rate / 100],  # 1.5% coupon
        day_count
    )
    bond.setPricingEngine(ql.DiscountingBondEngine(curve_handle))
    
    return bond.cleanPrice()




