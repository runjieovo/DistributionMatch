import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


# 分布名称映射到scipy.stats中的分布对象
distributions = {
    'Normal': stats.norm,
    'Exponential': stats.expon,
    'Bernoulli': stats.bernoulli,
    'Binomial': stats.binom,
    'Uniform': stats.uniform,
    'Pareto': stats.pareto,
    'T': stats.t,
    'LogNormal': stats.lognorm,
    'Weibull': stats.weibull_min,
    'Gamma': stats.gamma,
    'ChiSquared': stats.chi2,
    
    # 'GMM': GaussianMixture(n_components=3, random_state=0),
}

def read_flow(file_path):
    packet_count = []  # 用于保存包数量的列表
    total_bytes = []   # 用于保存总字节数的列表
    duration = []      # 用于保存持续时间的列表

    with open(file_path, 'r') as file:
        for line in file:
            # 将每行按空格分割，并转换为整数
            parts = [int(num) for num in line.split()]
            # 分别将数据添加到对应的列表中
            packet_count.append(parts[0])
            total_bytes.append(parts[1])
            duration.append(parts[2])

    return np.array(packet_count), np.array(total_bytes), np.array(duration)

def read_data(file_path):
    intervals = []
    frequencies = []
    with open(file_path, 'r') as file:
        for line in file:
            interval, frequency = map(int, line.split())
            intervals.extend([interval] * frequency)
    return np.array(intervals)


def fit_distributions(intervals):
    results = {}
    for name, dist in distributions.items():
        if name in ['Bernoulli', 'Binomial']:
            continue
        else:
            try:
                params = dist.fit(intervals)
                results[name] = params
            except Exception as e:
                print(f"Failed to fit {name}: {e}")

    p_estimate = np.mean(intervals) / np.max(intervals)
    results['Bernoulli'] = (p_estimate,)
    n_binom = np.max(intervals)
    results['Binomial'] = (n_binom, p_estimate)

    return results


def calculate_cross_entropy(intervals, distribution, params):
    try:
        if distribution in [stats.bernoulli, stats.binom]:
            pmf = distribution.pmf(intervals, *params)
        else:
            pdf = distribution.pdf(intervals, *params)
            
        values = np.clip(pmf if distribution in [stats.bernoulli, stats.binom] else pdf, 1e-10, 1)
        return -np.sum(np.log(values))
    except Exception as e:
        print(f"Failed to calculate entropy for {distribution.name}: {e}")
        return np.inf


def plot_best_fit_distribution(intervals, best_fit_name, best_fit_params, file):
    if best_fit_name == 'GMM':
        intervals = np.array(intervals).reshape(-1, 1)
        x_range = np.linspace(np.min(intervals), np.max(intervals), 1000).reshape(-1, 1)
        pdf = np.exp(best_fit_params.score_samples(x_range))
        plt.hist(intervals, bins=30, density=True, alpha=0.6, color='k', label='Original intervals')
        plt.plot(x_range, pdf, '-k', label='GMM fit')

        plt.title('Gaussian Mixture Model Fit')
        plt.xlabel('Data')
        plt.ylabel('Probability density')
        plt.legend()
    else:
        plt.figure(figsize=(10, 6))
        # Plot histogram of original data
        plt.hist(intervals, bins=30, density=True, alpha=0.6, color='g')

        # Generate points for pdf
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        dist = getattr(stats, distributions[best_fit_name].name)
        if distributions[best_fit_name] in [stats.bernoulli, stats.binom]:
            pmf = dist.pmf(np.round(x), *best_fit_params)
            plt.plot(np.round(x), pmf, 'k', linewidth=2)
        else:
            pdf = dist.pdf(x, *best_fit_params)
            plt.plot(x, pdf, 'k', linewidth=2)

        title = f"Fit results: {best_fit_name}"
        plt.title(title)
        plt.xlabel('Interval')
        plt.ylabel('Density')
        plt.show()
    plt.savefig('./img/' + file + '_' + best_fit_name + '.png')
    
def cal_GMM_entropy(intervals):
    intervals = np.array(intervals).reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=0)
    gmm.fit(intervals)
    log_pdf = gmm.score_samples(intervals)
    cel = -np.sum(log_pdf)
    return cel, gmm

def cal_cdf(interval, name, params):
    if name == 'GMM':
        def calculate(pos, params):
            return sum([w * norm.cdf(pos, mean, np.sqrt(cov)) for w, mean, cov in zip(params.weights_, params.means_.flatten(), params.covariances_.flatten())])
        # cdf = sum([w * norm.cdf(x_point, mean, np.sqrt(cov)) for w, mean, cov in zip(params.weights_, params.means_.flatten(), params.covariances_.flatten())])
        _min = interval.min()
        _max = interval.max()
        k = 100
        length = (_max - _min) // k
        interval_cdf = []
        lef = _min
        pre_cdf = 0
        for i in range(k - 1):
            cur_cdf = calculate(lef + length, params)
            interval_cdf.append((lef, lef + length, cur_cdf - pre_cdf))
            pre_cdf = cur_cdf
            lef += length
        interval_cdf.append((lef, _max, calculate(_max, params) - pre_cdf))
    else:
        distribution = distributions[name]
        _min = interval.min()
        _max = interval.max()
        k = 100
        length = (_max - _min) // k
        interval_cdf = []
        lef = _min
        pre_cdf = 0
        def calculate(pos, dis, params):
            return dis.cdf(pos, *params)
        for i in range(k - 1):
            cur_cdf = calculate(lef + length, distribution, params)
            interval_cdf.append((lef, lef + length, cur_cdf - pre_cdf))
            pre_cdf = cur_cdf
            lef += length
        interval_cdf.append((lef, _max, calculate(_max, distribution, params) - pre_cdf))
    with open('./intervalPos.txt', 'a') as file:
        file.write(name + '\n')
        for interval in interval_cdf:
            file.write("{}:{} {}\n".format(interval[0], interval[1], interval[2]))
        
    
def simulateFile(file):
    filePth = './output/{}.out'.format(file)
    if file in ['flowBytes', 'flowPktNum', 'flowDuration']:
        flowPktNum, flowBytes, flowDuration = read_flow('./output/fiveTupleInfo.out')
        intervals = locals().get(file)
    else:
        intervals = read_data(filePth)
    # GMM(intervals)
    fitted_distributions = fit_distributions(intervals)
    entropies = {}
    for name, params in fitted_distributions.items():
        dist = distributions[name]
        entropy = calculate_cross_entropy(intervals, dist, params)
        entropies[name] = entropy
    cel, gmm = cal_GMM_entropy(intervals)
    
    entropies['GMM'] = cel
    fitted_distributions['GMM'] = gmm
    print(entropies)
    best_fit = min(entropies, key=entropies.get)
    print(f"The best fitting distribution of {file} is {best_fit} with a cross entropy of {entropies[best_fit]:.2f}")
    with open('./matchResult.txt', 'a') as F:
        F.write(f"The best fitting distribution of {file} is {best_fit} with a cross entropy of {entropies[best_fit]:.2f}\n")
    cal_cdf(intervals, best_fit, fitted_distributions[best_fit])

    plot_best_fit_distribution(intervals, best_fit, fitted_distributions[best_fit], file)
    
if __name__ == '__main__':
    # file_path = './output/pktInterArrivalTime.out'
    targets = ['packetSize', 'pktInterArrivalTime', 'flowInterArrivalTime', 'burstDuration', 'burstPacketNum', 'burstByteCount', 'flowBytes', 'flowPktNum', 'flowDuration']
    # flow特征需要单独读取
    
    with open('./intervalPos.txt', 'wb') as file:
        # file.write('')
        pass
    with open('./matchResult.txt', 'wb') as file:
        pass
    for target in targets:
        simulateFile(target)