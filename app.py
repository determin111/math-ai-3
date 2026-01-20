from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp

app = Flask(__name__)


# 预处理函数：将用户输入的非标准数学写法转换为 SymPy 规范
def preprocess_expression(expr):
    # 替换常见符号
    expr = expr.replace('^', '**').replace('***', '**')
    # 处理类似 2x -> 2*x 的情况
    expr = sp.sympify(expr, locals={'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'pi': sp.pi})
    return expr


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    try:
        expr_str = data.get('expression', 'sin(x)')
        x0_val = float(data.get('x0', 2.0))

        x = sp.symbols('x')
        f_sym = preprocess_expression(expr_str)
        df_sym = sp.diff(f_sym, x)

        # 转换为 NumPy 可计算函数
        f_func = sp.lambdify(x, f_sym, 'numpy')
        df_func = sp.lambdify(x, df_sym, 'numpy')

        steps = []
        errors = []
        curr_x = x0_val

        # 牛顿迭代逻辑
        for i in range(20):
            # 关键修复：使用 float() 强制转换 SymPy 对象为 Python 原生 float
            fx = float(f_func(curr_x))
            dfx = float(df_func(curr_x))

            steps.append({"x": float(curr_x), "y": fx})
            if i > 0:
                errors.append(abs(curr_x - steps[-2]["x"]))

            if abs(dfx) < 1e-12: break  # 导数接近0，停止

            next_x = curr_x - fx / dfx
            if abs(next_x - curr_x) < 1e-8:
                steps.append({"x": float(next_x), "y": float(f_func(next_x))})
                break
            curr_x = next_x

        # 生成背景曲线数据 (范围根据解的位置动态调整)
        root = steps[-1]["x"]
        px = np.linspace(root - 5, root + 5, 400)
        # 确保 py 是 Python list
        py = [float(v) for v in f_func(px)]

        return jsonify({
            "success": True,
            "steps": steps,
            "errors": errors,
            "curve": {"x": px.tolist(), "y": py}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
