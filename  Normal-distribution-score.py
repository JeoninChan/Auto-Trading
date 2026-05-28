import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.widgets import Button

class MetricSlideshow:
    def __init__(self):
        # 1. 맷플롯립 스타일 및 창 크기 설정 (노트북 맞춤형 10x6.5)
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.ax = plt.subplots(figsize=(10, 6.5))
        
        # 하단 버튼 공간 확보를 위해 여백 조절
        plt.subplots_adjust(bottom=0.2)
        
        self.current_page = 0
        self.total_pages = 6
        
        # 2. 하단 내비게이션 버튼 UI 배치 [좌측여백, 아래여백, 가로폭, 세로높이]
        ax_prev = plt.axes([0.72, 0.04, 0.10, 0.06])
        ax_next = plt.axes([0.84, 0.04, 0.10, 0.06])
        
        self.btn_prev = Button(ax_prev, '◀ Prev', color='lightgray', hovercolor='aquamarine')
        self.btn_next = Button(ax_next, 'Next ▶', color='lightgray', hovercolor='aquamarine')
        
        # 버튼 클릭 이벤트 바인딩
        self.btn_prev.on_clicked(self.go_prev)
        self.btn_next.on_clicked(self.go_next)
        
        # 첫 번째 페이지 그리기
        self.draw_page()

    def go_prev(self, event):
        self.current_page = (self.current_page - 1) % self.total_pages
        self.draw_page()

    def go_next(self, event):
        self.current_page = (self.current_page + 1) % self.total_pages
        self.draw_page()

    def draw_page(self):
        # 이전 그림 초기화
        self.ax.clear()
        
        # 공통 Y축 설정 및 그리드 활성화
        self.ax.set_ylabel('Normalized Score (0 ~ 100)', fontsize=11, weight='bold')
        self.ax.set_ylim(-5, 105)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        
        page = self.current_page
        
        # ----------------------------------------------------
        # [SLIDE 1] 시가총액 (Market Cap)
        # ----------------------------------------------------
        if page == 0:
            market_caps = np.linspace(0.1, 40.0, 500)
            log_caps = np.log10(market_caps * 1e12)
            log_min, log_max = np.log10(0.5 * 1e12), np.log10(28.0 * 1e12)
            sigma_cap = 0.15
            
            scores_cap = np.where(
                (log_caps >= log_min) & (log_caps <= log_max), 100.0,
                np.where(log_caps < log_min,
                         100.0 * np.exp(-((log_caps - log_min) ** 2) / (2 * sigma_cap ** 2)),
                         100.0 * np.exp(-((log_caps - log_max) ** 2) / (2 * sigma_cap ** 2)))
            )
            self.ax.plot(market_caps, scores_cap, color='darkgreen', linewidth=2.5)
            self.ax.axvspan(0.5, 28.0, color='green', alpha=0.1, label='100pt Plateau (5000B ~ 28T)')
            self.ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='5,000억 원 (Boundary)')
            self.ax.axvline(28.0, color='red', linestyle='--', alpha=0.7, label='28조 원 (Boundary)')
            self.ax.set_title('[1 / 6] Market Cap (Flat-top Plateau Model)', fontsize=13, weight='bold', pad=15)
            self.ax.set_xlabel('Market Cap (Trillion KRW)', fontsize=11)
            self.ax.set_xlim(0, 40)

        # ----------------------------------------------------
        # [SLIDE 2] 부채비율 (Debt to Equity)
        # ----------------------------------------------------
        elif page == 1:
            de_ratio = np.linspace(0, 250, 500)
            scores_de = (1 - stats.norm.cdf((de_ratio - 100) / 40)) * 100
            
            self.ax.plot(de_ratio, scores_de, color='crimson', linewidth=2.5)
            self.ax.axvline(100, color='black', linestyle='--', alpha=0.7, label='Old Cutoff (100% -> 50pt)')
            self.ax.set_title('[2 / 6] Debt to Equity Ratio (Lower is Better)', fontsize=13, weight='bold', pad=15)
            self.ax.set_xlabel('D/E Ratio (%)', fontsize=11)

        # ----------------------------------------------------
        # [SLIDE 3] 매출총이익률 (Gross Profit Margin)
        # ----------------------------------------------------
        elif page == 2:
            gp_margin = np.linspace(0, 80, 500)
            scores_gp = stats.norm.cdf((gp_margin - 10) / 15) * 100
            
            self.ax.plot(gp_margin, scores_gp, color='royalblue', linewidth=2.5)
            self.ax.axvline(10, color='black', linestyle='--', alpha=0.7, label='Old Cutoff (10% -> 50pt)')
            self.ax.set_title('[3 / 6] Gross Profit Margin (Higher is Better)', fontsize=13, weight='bold', pad=15)
            self.ax.set_xlabel('Gross Margin (%)', fontsize=11)

        # ----------------------------------------------------
        # [SLIDE 4] 순이익률 (Net Profit Margin)
        # ----------------------------------------------------
        elif page == 3:
            np_margin = np.linspace(-5, 40, 500)
            scores_np = stats.norm.cdf((np_margin - 5) / 10) * 100
            
            self.ax.plot(np_margin, scores_np, color='purple', linewidth=2.5)
            self.ax.axvline(5, color='black', linestyle='--', alpha=0.7, label='Old Cutoff (5% -> 50pt)')
            self.ax.set_title('[4 / 6] Net Profit Margin (Higher is Better)', fontsize=13, weight='bold', pad=15)
            self.ax.set_xlabel('Net Margin (%)', fontsize=11)

        # ----------------------------------------------------
        # [SLIDE 5] 매출성장률 (Revenue Growth)
        # ----------------------------------------------------
        elif page == 4:
            rev_growth = np.linspace(-10, 60, 500)
            scores_rev = stats.norm.cdf((rev_growth - 20) / 15) * 100
            
            self.ax.plot(rev_growth, scores_rev, color='darkorange', linewidth=2.5)
            self.ax.axvline(20, color='black', linestyle='--', alpha=0.7, label='Old Cutoff (20% -> 50pt)')
            self.ax.set_title('[5 / 6] Revenue Growth (Higher is Better)', fontsize=13, weight='bold', pad=15)
            self.ax.set_xlabel('Revenue Growth (%)', fontsize=11)

        # ----------------------------------------------------
        # [SLIDE 6] PEG 비율 (Price/Earnings-to-Growth)
        # ----------------------------------------------------
        elif page == 5:
            peg_ratio = np.linspace(0.1, 2.5, 500)
            scores_peg = np.where(peg_ratio <= 0.5, 100.0, np.maximum(0.0, 100.0 - 50.0 * (peg_ratio - 0.5)))
            
            self.ax.plot(peg_ratio, scores_peg, color='teal', linewidth=2.5)
            self.ax.axvline(0.5, color='blue', linestyle=':', alpha=0.8, label='PEG <= 0.5 (100pt)')
            self.ax.axvline(1.0, color='purple', linestyle=':', alpha=0.8, label='PEG 1.0 (75pt)')
            self.ax.axvline(1.5, color='red', linestyle='--', alpha=0.8, label='PEG 1.5 (50pt Cutoff)')
            self.ax.set_title('[6 / 6] PEG Ratio (Valuation Linear-Mapping)', fontsize=13, weight='bold', pad=15)
            self.ax.set_xlabel('PEG Ratio', fontsize=11)
            self.ax.set_xlim(0.1, 2.0)

        self.ax.legend(loc='upper right', fontsize=10)
        plt.draw()

# 스크립트 실행
if __name__ == '__main__':
    slideshow = MetricSlideshow()
    plt.show()