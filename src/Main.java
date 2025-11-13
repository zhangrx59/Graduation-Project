import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.sql.*;

// 程序入口
public class Main {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(LoginFrame::new);
    }
}

/**
 * 简单 UI 工具：把图片淡化（叠加一层半透明白色）
 */
class UIUtils {
    /**
     * 从 classpath 加载图片，并进行淡化处理
     * @param resourcePath 资源路径，例如 "/background.jpg"
     * @param whiteAlpha   白色叠加透明度（0~1，0.4~0.7 比较合适）
     */
    public static ImageIcon createDimmedIcon(String resourcePath, float whiteAlpha) {
        java.net.URL url = UIUtils.class.getResource(resourcePath);
        if (url == null) {
            // 找不到资源时，返回一个空的图标避免 NPE
            return new ImageIcon();
        }
        ImageIcon icon = new ImageIcon(url);
        Image src = icon.getImage();
        int w = src.getWidth(null);
        int h = src.getHeight(null);
        if (w <= 0 || h <= 0) {
            return icon;
        }

        BufferedImage buf = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = buf.createGraphics();
        // 先绘制原始图片
        g2.drawImage(src, 0, 0, null);
        // 再叠加一层半透明白色，达到“淡化”的效果
        g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, whiteAlpha));
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, w, h);
        g2.dispose();

        return new ImageIcon(buf);
    }
}

// 登录窗口
class LoginFrame extends JFrame implements ActionListener {

    private JTextField usernameField;
    private JPasswordField passwordField;
    private JLabel statusLabel;
    private JButton loginButton;
    private JButton registerButton;
    private JButton exitButton;   // 登录界面的退出程序按钮

    public LoginFrame() {
        setTitle("登录窗口");
        setSize(1920, 1280);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null); // 窗口居中
        setResizable(false);

        // === 1. 设置背景图片（淡化版） ===
        ImageIcon bgIcon = UIUtils.createDimmedIcon("/background.jpg", 0.45f);
        JLabel bgLabel = new JLabel(bgIcon);
        setContentPane(bgLabel);
        bgLabel.setLayout(null);

        // 统一文字颜色：黑色
        Color textColor = Color.BLACK;

        // === 2. 定义字体（加粗 + 变大） ===
        Font titleFont = new Font("微软雅黑", Font.BOLD, 48);  // 欢迎语更大
        Font labelFont = new Font("微软雅黑", Font.BOLD, 30);   // 标签：稍微加大
        Font fieldFont = new Font("微软雅黑", Font.PLAIN, 24);   // 输入框
        Font buttonFont = new Font("微软雅黑", Font.BOLD, 26);  // 按钮
        Font statusFont = new Font("微软雅黑", Font.BOLD, 22);  // 状态提示

        // === 2.1 欢迎语（登录界面顶部，居中） ===
        JLabel welcomeLabel = new JLabel("欢迎使用医疗大模型诊断系统");
        welcomeLabel.setFont(titleFont);
        welcomeLabel.setForeground(textColor);
        welcomeLabel.setHorizontalAlignment(SwingConstants.CENTER);
        welcomeLabel.setBounds(0, 120, 1920, 80);
        bgLabel.add(welcomeLabel);

        // === 3. 表单区域 ===
        int formLeftX = 760;   // 整个表单左边起点 X
        int labelWidth = 140;
        int labelHeight = 40;
        int fieldWidth = 320;
        int fieldHeight = 40;

        // 用户名标签
        JLabel userLabel = new JLabel("用户名：");
        userLabel.setForeground(textColor);
        userLabel.setFont(labelFont);
        userLabel.setBounds(formLeftX, 500, labelWidth, labelHeight);
        bgLabel.add(userLabel);

        // 用户名输入框
        usernameField = new JTextField();
        usernameField.setFont(fieldFont);
        usernameField.setForeground(textColor);
        usernameField.setOpaque(false);
        usernameField.setBounds(formLeftX + labelWidth + 20, 500, fieldWidth, fieldHeight);
        bgLabel.add(usernameField);

        // 密码标签
        JLabel passLabel = new JLabel("密码：");
        passLabel.setForeground(textColor);
        passLabel.setFont(labelFont);
        passLabel.setBounds(formLeftX, 570, labelWidth, labelHeight);
        bgLabel.add(passLabel);

        // 密码输入框
        passwordField = new JPasswordField();
        passwordField.setFont(fieldFont);
        passwordField.setForeground(textColor);
        passwordField.setOpaque(false);
        passwordField.setBounds(formLeftX + labelWidth + 20, 570, fieldWidth, fieldHeight);
        bgLabel.add(passwordField);

        // 登录按钮
        loginButton = new JButton("登录");
        loginButton.setFont(buttonFont);
        loginButton.setForeground(textColor);
        loginButton.setBounds(formLeftX + labelWidth + 20, 640, 160, 50);
        loginButton.setContentAreaFilled(false);
        loginButton.setBorderPainted(true);
        loginButton.addActionListener(this);
        bgLabel.add(loginButton);

        // 注册按钮
        registerButton = new JButton("注册");
        registerButton.setFont(buttonFont);
        registerButton.setForeground(textColor);
        registerButton.setBounds(formLeftX + labelWidth + 20 + 200, 640, 160, 50);
        registerButton.setContentAreaFilled(false);
        registerButton.setBorderPainted(true);
        registerButton.addActionListener(e -> openRegisterWindow());
        bgLabel.add(registerButton);

        // 登录界面：退出程序按钮
        exitButton = new JButton("退出程序");
        exitButton.setFont(buttonFont);
        exitButton.setForeground(textColor);
        exitButton.setBounds(formLeftX + labelWidth + 20 + 100, 710, 200, 50);
        exitButton.setContentAreaFilled(false);
        exitButton.setBorderPainted(true);
        exitButton.addActionListener(e -> System.exit(0));
        bgLabel.add(exitButton);

        // 状态提示文字
        statusLabel = new JLabel("");
        statusLabel.setForeground(textColor);
        statusLabel.setFont(statusFont);
        statusLabel.setBounds(formLeftX, 780, 800, 35);
        bgLabel.add(statusLabel);

        setVisible(true);
    }

    // 打开注册窗口
    private void openRegisterWindow() {
        new RegisterFrame(this);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        String inputUser = usernameField.getText();
        String inputPass = new String(passwordField.getPassword());

        if (inputUser.isEmpty() || inputPass.isEmpty()) {
            statusLabel.setText("用户名或密码不能为空！");
            return;
        }

        boolean ok = DBHelper.validateLogin(inputUser, inputPass);
        if (ok) {
            statusLabel.setText("登录成功！");
            JOptionPane.showMessageDialog(this, "登录成功！");
            // 登录成功后，打开主界面（显示淡化后的 pic2.jpg），并关闭登录窗口
            new MainAppFrame();
            dispose();
        } else {
            statusLabel.setText("用户名或密码错误！");
        }
    }
}

// 主界面窗口（登录成功后显示淡化的 pic2.jpg，包含欢迎语、退出登录、退出程序）
class MainAppFrame extends JFrame implements ActionListener {

    private JButton logoutButton;
    private JButton exitButton;

    public MainAppFrame() {
        setTitle("医疗大模型诊断系统 - 主界面");
        setSize(1920, 1280);
        setLocationRelativeTo(null);
        setResizable(false);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // 设置主界面背景图：淡化版 pic2.jpg
        ImageIcon bgIcon = UIUtils.createDimmedIcon("/background2.jpg", 0.45f);
        JLabel bgLabel = new JLabel(bgIcon);
        setContentPane(bgLabel);
        bgLabel.setLayout(null);

        Color textColor = Color.BLACK;
        Font titleFont = new Font("微软雅黑", Font.BOLD, 48);
        Font buttonFont = new Font("微软雅黑", Font.BOLD, 26);

        // 顶部欢迎语（居中）
        JLabel welcomeLabel = new JLabel("欢迎使用医疗大模型诊断系统");
        welcomeLabel.setFont(titleFont);
        welcomeLabel.setForeground(textColor);
        welcomeLabel.setHorizontalAlignment(SwingConstants.CENTER);
        welcomeLabel.setBounds(0, 120, 1920, 80);
        bgLabel.add(welcomeLabel);

        // 退出登录按钮
        logoutButton = new JButton("退出登录");
        logoutButton.setFont(buttonFont);
        logoutButton.setForeground(textColor);
        logoutButton.setBounds(760, 700, 160, 50);
        logoutButton.setContentAreaFilled(false);
        logoutButton.setBorderPainted(true);
        logoutButton.addActionListener(this);
        bgLabel.add(logoutButton);

        // 退出程序按钮
        exitButton = new JButton("退出程序");
        exitButton.setFont(buttonFont);
        exitButton.setForeground(textColor);
        exitButton.setBounds(1000, 700, 160, 50);
        exitButton.setContentAreaFilled(false);
        exitButton.setBorderPainted(true);
        exitButton.addActionListener(this);
        bgLabel.add(exitButton);

        setVisible(true);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        Object src = e.getSource();
        if (src == logoutButton) {
            // 退出登录：关闭当前主界面，回到登录窗口
            dispose();
            new LoginFrame();
        } else if (src == exitButton) {
            // 退出程序：直接结束 JVM
            System.exit(0);
        }
    }
}

// 注册窗口
class RegisterFrame extends JDialog implements ActionListener {

    private JTextField usernameField;
    private JPasswordField passwordField;
    private JPasswordField confirmField;
    private JLabel statusLabel;
    private JButton submitButton;

    public RegisterFrame(JFrame parent) {
        super(parent, "用户注册", true); // 模态对话框
        setSize(600, 400);
        setLocationRelativeTo(parent);
        setLayout(null);

        Color textColor = Color.BLACK;
        Font labelFont = new Font("微软雅黑", Font.BOLD, 22);
        Font fieldFont = new Font("微软雅黑", Font.PLAIN, 18);
        Font buttonFont = new Font("微软雅黑", Font.BOLD, 20);
        Font statusFont = new Font("微软雅黑", Font.BOLD, 16);

        int leftX = 80;
        int labelWidth = 120;
        int height = 35;
        int fieldWidth = 260;

        // 用户名
        JLabel userLabel = new JLabel("用户名：");
        userLabel.setFont(labelFont);
        userLabel.setForeground(textColor);
        userLabel.setBounds(leftX, 60, labelWidth, height);
        add(userLabel);

        usernameField = new JTextField();
        usernameField.setFont(fieldFont);
        usernameField.setForeground(textColor);
        usernameField.setOpaque(false);
        usernameField.setBounds(leftX + labelWidth + 10, 60, fieldWidth, height);
        add(usernameField);

        // 密码
        JLabel passLabel = new JLabel("密码：");
        passLabel.setFont(labelFont);
        passLabel.setForeground(textColor);
        passLabel.setBounds(leftX, 110, labelWidth, height);
        add(passLabel);

        passwordField = new JPasswordField();
        passwordField.setFont(fieldFont);
        passwordField.setForeground(textColor);
        passwordField.setOpaque(false);
        passwordField.setBounds(leftX + labelWidth + 10, 110, fieldWidth, height);
        add(passwordField);

        // 确认密码
        JLabel confirmLabel = new JLabel("确认密码：");
        confirmLabel.setFont(labelFont);
        confirmLabel.setForeground(textColor);
        confirmLabel.setBounds(leftX, 160, labelWidth + 40, height);
        add(confirmLabel);

        confirmField = new JPasswordField();
        confirmField.setFont(fieldFont);
        confirmField.setForeground(textColor);
        confirmField.setOpaque(false);
        confirmField.setBounds(leftX + labelWidth + 50, 160, fieldWidth, height);
        add(confirmField);

        // 提交按钮
        submitButton = new JButton("提交注册");
        submitButton.setFont(buttonFont);
        submitButton.setForeground(textColor);
        submitButton.setBounds(leftX + 80, 220, 200, 40);
        submitButton.setContentAreaFilled(false);
        submitButton.setBorderPainted(true);
        submitButton.addActionListener(this);
        add(submitButton);

        // 状态提示
        statusLabel = new JLabel("");
        statusLabel.setFont(statusFont);
        statusLabel.setForeground(textColor);
        statusLabel.setBounds(leftX, 280, 450, 30);
        add(statusLabel);

        setVisible(true);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        String user = usernameField.getText().trim();
        String pass = new String(passwordField.getPassword());
        String confirm = new String(confirmField.getPassword());

        if (user.isEmpty() || pass.isEmpty() || confirm.isEmpty()) {
            statusLabel.setText("所有字段都不能为空！");
            return;
        }
        if (!pass.equals(confirm)) {
            statusLabel.setText("两次输入的密码不一致！");
            return;
        }

        boolean ok = DBHelper.registerUser(user, pass);
        if (ok) {
            JOptionPane.showMessageDialog(this, "注册成功，可以用新账号登录了！");
            dispose(); // 关闭注册窗口
        } else {
            statusLabel.setText("注册失败：用户名可能已存在");
        }
    }
}

// 数据库帮助类：SQLite + SQL 查询
class DBHelper {

    private static final String DB_URL = "jdbc:sqlite:users.db";

    static {
        // 1. 加载 SQLite 驱动
        try {
            Class.forName("org.sqlite.JDBC");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(null,
                    "未找到 SQLite JDBC 驱动，请确认已添加 sqlite-jdbc.jar",
                    "数据库错误",
                    JOptionPane.ERROR_MESSAGE);
        }

        // 2. 初始化数据库（建表）
        initDatabase();
    }

    // 获取数据库连接
    private static Connection getConnection() throws SQLException {
        return DriverManager.getConnection(DB_URL);
    }

    // 创建 users 表（如果不存在）
    private static void initDatabase() {
        String sql = "CREATE TABLE IF NOT EXISTS users (" +
                "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                "username TEXT UNIQUE NOT NULL," +
                "password TEXT NOT NULL" +
                ")";
        try (Connection conn = getConnection();
             Statement stmt = conn.createStatement()) {
            stmt.execute(sql);
        } catch (SQLException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(null,
                    "初始化数据库失败：" + e.getMessage(),
                    "数据库错误",
                    JOptionPane.ERROR_MESSAGE);
        }
    }

    // 注册新用户（INSERT SQL）
    public static boolean registerUser(String username, String password) {
        String sql = "INSERT INTO users(username, password) VALUES (?, ?)";
        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, username);
            ps.setString(2, password);
            ps.executeUpdate();
            return true;
        } catch (SQLException e) {
            System.err.println("注册失败：" + e.getMessage());
            return false;
        }
    }

    // 校验登录（SELECT + WHERE）
    public static boolean validateLogin(String username, String password) {
        String sql = "SELECT COUNT(*) FROM users WHERE username = ? AND password = ?";
        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setString(1, username);
            ps.setString(2, password);

            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) {
                    int count = rs.getInt(1);
                    return count > 0;
                }
            }
        } catch (SQLException e) {
            System.err.println("登录查询失败：" + e.getMessage());
        }
        return false;
    }
}
