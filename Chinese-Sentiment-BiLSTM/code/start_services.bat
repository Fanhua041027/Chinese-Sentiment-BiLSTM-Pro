@echo off
chcp 65001 >nul
echo ============================================================
echo 启动情感分析后端服务
echo ============================================================
echo.

echo [1/3] 启动 DeepSeek API 服务（端口 5000）...
start "DeepSeek API 服务" python deepseek_api.py
timeout /t 3 /nobreak >nul
echo.

echo [2/3] 启动本地模型服务（端口 5001）...
start "本地模型服务" python local_model_api.py
timeout /t 8 /nobreak >nul
echo.

echo [3/3] 启动 Web 服务器（端口 8000）...
start "Web 服务器" python -m http.server 8000
timeout /t 3 /nobreak >nul
echo.

echo ============================================================
echo ✅ 所有服务已启动！
echo ============================================================
echo.
echo 访问地址：
echo   - 情感分析页面：http://localhost:8000/advanced_app.html
echo   - 服务状态测试：test_services.html（在浏览器中打开）
echo.
echo 服务说明：
echo   - DeepSeek API 服务：端口 5000
echo   - 本地模型服务：端口 5001
echo   - Web 服务器：端口 8000
echo.
echo 按 Ctrl+C 或关闭窗口停止服务
echo ============================================================
