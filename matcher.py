import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from collections import Counter


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
        element_counts = Counter(intervals)
        # 提取元素和它们出现的次数到两个新数组
        elements = list(element_counts.keys())
        counts = list(element_counts.values())
        if distribution in [stats.bernoulli, stats.binom]:
            pmf = distribution.pmf(elements, *params)
            pmf = [weight * count for weight, count in zip(pmf, counts)]
        else:
            pdf = distribution.pdf(elements, *params)
            pdf = [weight * count for weight, count in zip(pdf, counts)]
        
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
    ii = [int(i) for i in intervals if i < 1400]
    gap = len(intervals) - len(ii)
    gap = int(gap * 0.9)
    for i in range(gap):
        ii.append(1400)
    gmm = GaussianMixture(n_components=3, random_state=2)
    # , means_init=[[10], [1400], [300]]
    # intervals = ii
    # intervals = np.array(intervals).reshape(-1, 1)
    gmm.fit(intervals)
    
    element_counts = Counter(list(intervals.reshape(-1)))
    # 提取元素和它们出现的次数到两个新数组
    elements = list(element_counts.keys())
    counts = list(element_counts.values())
    log_pdf = gmm.score_samples(np.array(elements).reshape(-1, 1))
    log_pdf = [weight * count for weight, count in zip(log_pdf, counts)]
    # log_pdf = np.s
    cel = -np.sum(log_pdf)
    return cel, gmm

def cal_cdf(interval, name, params, intervalName, k=100):
    if name == 'GMM':
        def calculate(pos, params):
            return sum([w * norm.cdf(pos, mean, np.sqrt(cov)) for w, mean, cov in zip(params.weights_, params.means_.flatten(), params.covariances_.flatten())])
        # cdf = sum([w * norm.cdf(x_point, mean, np.sqrt(cov)) for w, mean, cov in zip(params.weights_, params.means_.flatten(), params.covariances_.flatten())])
        _min = interval.min()
        _max = interval.max()
        length = (_max - _min + k - 1) // k
        interval_cdf = []
        lef = _min
        pre_cdf = calculate(lef, params)
        posList = []
        while lef < _max:
            posList.append(lef)
            lef += length
        posList.append(_max)
        for i in range(1, len(posList)):
            cur_cdf = calculate(posList[i], params)
            interval_cdf.append((int(posList[i - 1] + 1), int(posList[i]), cur_cdf - pre_cdf))
            pre_cdf = cur_cdf
    else:
        distribution = distributions[name]
        if intervalName == 'pktInterArrivalTime':
            _min = 0
            burst = 1e4
            _max = interval.max()
            small_part = np.arange(_min, burst, 1, dtype=int)
            small_part = list(small_part)
            length = (_max - burst + k - 1) // k
            lef = burst + length
            while lef < _max:
                small_part.append(lef)
                lef += length
            small_part.append(_max)
            posList = small_part # 前burst个点都是整数分割，后面分为1000份
            interval_cdf = []
            def calculate(pos, dis, params):
                return dis.cdf(pos, *params)
            pre_cdf = calculate(0, distribution, params)
            interval_cdf.append((0, 0, pre_cdf))
            for i in range(1, len(posList)):
                cur_cdf = calculate(posList[i], distribution, params)
                interval_cdf.append((int(posList[i - 1] + 1), int(posList[i]), cur_cdf - pre_cdf))
                pre_cdf = cur_cdf
        else:
            def calculate(pos, dis, params):
                return dis.cdf(pos, *params)
            _min = interval.min()
            _max = interval.max()
            length = (_max - _min + k - 1) // k
            interval_cdf = []
            lef = _min
            posList = []
            pre_cdf = calculate(lef, distribution, params)
            while lef < _max:
                posList.append(lef)
                lef += length
            posList.append(_max) 
            for i in range(1, len(posList)):
                cur_cdf = calculate(posList[i], distribution, params)
                interval_cdf.append((int(posList[i - 1] + 1), int(posList[i]), cur_cdf - pre_cdf))
                pre_cdf = cur_cdf
    with open('./intervalPos.txt', 'a') as file:
        file.write(intervalName + '\n')
        for interval in interval_cdf:
            file.write("{}:{} {}\n".format(interval[0], interval[1], interval[2]))
        file.write("#\n")
        
    
def simulateFile(file):
    filePth = './bilibili_output/{}.out'.format(file)
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
    if cel >= 0:
        entropies['GMM'] = cel
        fitted_distributions['GMM'] = gmm
    print(entropies)
    best_fit = min(entropies, key=entropies.get)
    print(f"The best fitting distribution of {file} is {best_fit} with a cross entropy of {entropies[best_fit]:.2f}")
    with open('./matchResult.txt', 'a') as F:
        if best_fit == 'GMM':
            tt = fitted_distributions[best_fit]
            F.write("GMM: [({}, {}), ({}, {}), ({}, {})]\n".format(tt.covariances_[0,0,0], tt.means_[0,0], tt.covariances_[1,0,0], tt.means_[1,0], tt.covariances_[2,0,0], tt.means_[2,0]))
        else:
            F.write('{}: {}\n'.format(best_fit, fitted_distributions[best_fit]))
        F.write(f"The best fitting distribution of {file} is {best_fit} with a cross entropy of {entropies[best_fit]:.2f}\n")
    cal_cdf(intervals, best_fit, fitted_distributions[best_fit], file, k=100)
    plot_best_fit_distribution(intervals, best_fit, fitted_distributions[best_fit], file)
    
if __name__ == '__main__':
    # file_path = './output/pktInterArrivalTime.out'
    # targets = ['packetSize', 'pktInterArrivalTime', 'flowInterArrivalTime', 'burstDuration', 'burstPacketNum', 'burstByteCount', 'flowBytes', 'flowPktNum', 'flowDuration']
    # targets = ['burstDuration']
    targets = ['packetSize']
    # flow特征需要单独读取
    
    with open('./intervalPos.txt', 'wb') as file:
        # file.write('')
        pass
    with open('./matchResult.txt', 'wb') as file:
        pass
    for target in targets:
        simulateFile(target)