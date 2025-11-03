import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import re

class IRSpectrumProcessor:
    def __init__(self, file_path):
        """
        初始化类对象并解析文件
        :param file_path: 红外光谱计算结果文件路径
        """
        self.file_path = file_path
        self.freqs, self.intensities = self._parse_ir_data()

    def _parse_ir_data(self):
        """
        从文本文件中提取频率和强度（km/mol）
        """
        freqs = []
        intensities = []

        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            match = re.match(r'\s*\d+:\s+([\d\.]+)\s+[\d\.Ee-]+\s+([\d\.Ee-]+)', line)
            if match:
                freq = float(match.group(1))
                intensity = float(match.group(2))
                freqs.append(freq)
                intensities.append(intensity)

        return np.array(freqs), np.array(intensities)

    def _voigt_profile(self, x, center, intensity, sigma, gamma):
        """
        Voigt型展宽函数
        """
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return intensity * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    def generate_spectrum(self, x_range=(0, 4000), resolution=1, sigma=2, gamma=2, normalize=True):
        """
        展宽并生成红外光谱数据
        :param x_range: 频率范围 (start, end)，单位 cm^-1
        :param resolution: 频率分辨率
        :param sigma: 高斯展宽参数
        :param gamma: 洛伦兹展宽参数
        :param normalize: 是否归一化强度
        :return: (x轴, 展宽后光谱)
        """
        x = np.arange(x_range[0], x_range[1], resolution)
        spectrum = np.zeros_like(x, dtype=np.float64)

        for freq, inten in zip(self.freqs, self.intensities):
            spectrum += self._voigt_profile(x, freq, inten, sigma, gamma)

        if normalize and spectrum.max() != 0:
            spectrum /= spectrum.max()

        self.x = x
        self.spectrum = spectrum
        return x, spectrum

    def plot_spectrum(self, invert_xaxis=True):
        """
        绘制光谱图
        :param invert_xaxis: 是否倒置横坐标（常用于IR谱）
        """
        if not hasattr(self, 'spectrum'):
            raise ValueError("请先调用 generate_spectrum() 生成光谱")

        plt.figure(figsize=(10, 4))
        plt.plot(self.x, self.spectrum, color='black')
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Intensity (a.u.)")
        plt.title("IR Spectrum (Voigt Broadened)")
        if invert_xaxis:
            plt.gca().invert_xaxis()
        plt.tight_layout()
        # plt.show()
        plt.savefig('spectrum.png', dpi=300)
        plt.close()

    def export_spectrum(self, output_path='spectrum.csv'):
        """
        将展宽后的光谱保存为 CSV 文件
        """
        if not hasattr(self, 'spectrum'):
            raise ValueError("请先调用 generate_spectrum() 生成光谱")
        np.savetxt(output_path, np.column_stack((self.x, self.spectrum)), delimiter=",", header="Wavenumber,Intensity", comments='')
