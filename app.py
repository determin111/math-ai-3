from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# 1. 牛顿迭代法求根
@app.route('/newton', methods=['POST'])
def newton():
    data = request.json
    expr_str = data.get('expression', 'x**2 - 2')
    x0 = float(data.get('x0', 1.0))

    x = sp.symbols('x')
    f = sp.parse_expr(expr_str)
    df = sp.diff(f, x)

    f_func = sp.lambdify(x, f, 'numpy')
    df_func = sp.lambdify(x, df, 'numpy')

    steps = []
    curr_x = x0
    for i in range(10):  # 演示前10步
        fx = f_func(curr_x)
        dfx = df_func(curr_x)
        if abs(dfx) < 1e-10: break

        next_x = curr_x - fx / dfx
        steps.append({"iteration": i, "x": float(curr_x), "y": float(fx)})
        if abs(next_x - curr_x) < 1e-6: break
        curr_x = next_x

    return jsonify({"success": True, "steps": steps})


# 2. 最小二乘法拟合
@app.route('/least_squares', methods=['POST'])
def least_squares():
    data = request.json
    points = data.get('points', [[0, 1], [1, 2], [2, 3]])  # 用户输入的数据点
    pts = np.array(points)
    x_data = pts[:, 0]
    y_data = pts[:, 1]

    # 线性拟合 y = ax + b
    A = np.vstack([x_data, np.ones(len(x_data))]).T
    a, b = np.linalg.lstsq(A, y_data, rcond=None)[0]

    return jsonify({
        "success": True,
        "slope": float(a),
        "intercept": float(b),
        "x_range": [float(min(x_data) - 1), float(max(x_data) + 1)]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)