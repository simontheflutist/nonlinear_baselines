# %%
import nonlinear_benchmarks as nlb
from MDC import np, plt


def plot_split(name, data):
    print(name)
    train, val, test = data
    print("train", train.y.shape)
    print("val", val.y.shape)
    for i, te in enumerate(test):
        print(f"test_{i}", te.y.shape)
    lt, lv = len(train), len(val)
    ltes = [len(te) for te in test]
    xt = np.arange(lt + lv + sum(ltes))
    plt.figure()
    plt.plot(xt[:lt], train.y)
    plt.plot(xt[lt : lt + lv], val.y)
    st = lt + lv
    for te, lte in zip(test, ltes):
        plt.plot(xt[st : st + lte], te.y, ls="--")
        st += lte


# %% CascadedTanks

train, test = nlb.Cascaded_Tanks()
cut = 700
CT = train[:cut], train[cut:], (test,)
# plot_split("CT", CT)

# %% Silverbox

train_val, test = nlb.Silverbox()
cut = len(train_val) // 2
SB = train_val[:cut], train_val[cut:], test
# plot_split("SB", SB)


# %% EMPS
train_val, test = nlb.EMPS()
cut = len(train_val) // 2
EMPS = train_val[:cut], train_val[cut:], (test,)

# plot_split("EMPS", EMPS)


# %% CED1

(train_val, _), (test, _) = nlb.CED()
cut = len(train_val) // 2
CED1 = train_val[:cut], train_val[cut:], (test,)

# plot_split("CED1", CED1)

# %% CED2

(_, train_val), (_, test) = nlb.CED()
cut = len(train_val) // 2
CED2 = train_val[:cut], train_val[cut:], (test,)


# plot_split("CED2", CED2)

# %% WH

train_val, test = nlb.WienerHammerBenchMark()
cut = len(train_val) // 2
WH = train_val[:cut], train_val[cut:], (test,)
# plot_split("WH", WH)

# %% all data

benchmarks_tvt = {
    "SB": SB,
    "EMPS": EMPS,
    "WH": WH,
    "CT": CT,
    "CED1": CED1,
    "CED2": CED2,
}
