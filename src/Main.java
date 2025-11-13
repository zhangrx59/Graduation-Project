import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.*;

// 程序入口
public class Main {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(LoginFrame::new);
    }
}

// 登录窗口
class LoginFrame extends JFrame implements ActionListener {

    private JTextField usernameField;
    private JPasswordField passwordField;
    private JLabel statusLabel;
    private JButton loginButton;
    private JButton registerButton;

    public LoginFrame() {
        setTitle("登录窗口");
        setSize(1920, 1280);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null); // 窗口居中
        setResizable(false);

        // === 1. 设置背景图片 ===
        // 注意：background.jpg 放在「工程根目录」，和 src 同级
        ImageIcon bgIcon = new ImageIcon(
                LoginFrame.class.getResource("/background.jpg")
        );
        JLabel bgLabel = new JLabel(bgIcon);
        setContentPane(bgLabel);

        // 使用绝对布局，手动摆放控件
        bgLabel.setLayout(null);

        // === 2. 定义字体（加粗 + 变大） ===
        Font labelFont = new Font("微软雅黑", Font.BOLD, 28);   // 标签：加粗
        Font fieldFont = new Font("微软雅黑", Font.PLAIN, 24);   // 输入框
        Font buttonFont = new Font("微软雅黑", Font.BOLD, 24);  // 按钮
        Font statusFont = new Font("微软雅黑", Font.BOLD, 20);  // 状态提示

        // === 3. 计算大致中间位置（手动调好的坐标）===
        int formLeftX = 760;   // 整个表单左边起点 X
        int labelWidth = 120;
        int labelHeight = 40;
        int fieldWidth = 320;
        int fieldHeight = 40;

        // 用户名标签
        JLabel userLabel = new JLabel("用户名：");
        userLabel.setForeground(Color.BLACK);
        userLabel.setFont(labelFont);
        userLabel.setBounds(formLeftX, 500, labelWidth, labelHeight);
        bgLabel.add(userLabel);

        // 用户名输入框
        usernameField = new JTextField();
        usernameField.setFont(fieldFont);
        usernameField.setBounds(formLeftX + labelWidth + 20, 500, fieldWidth, fieldHeight);
        bgLabel.add(usernameField);

        // 密码标签
        JLabel passLabel = new JLabel("密码：");
        passLabel.setForeground(Color.BLACK);
        passLabel.setFont(labelFont);
        passLabel.setBounds(formLeftX, 570, labelWidth, labelHeight);
        bgLabel.add(passLabel);

        // 密码输入框
        passwordField = new JPasswordField();
        passwordField.setFont(fieldFont);
        passwordField.setBounds(formLeftX + labelWidth + 20, 570, fieldWidth, fieldHeight);
        bgLabel.add(passwordField);

        // 登录按钮
        loginButton = new JButton("登录");
        loginButton.setFont(buttonFont);
        loginButton.setBounds(formLeftX + labelWidth + 20, 640, 160, 45);
        loginButton.addActionListener(this);
        bgLabel.add(loginButton);

        // 注册按钮
        registerButton = new JButton("注册");
        registerButton.setFont(buttonFont);
        registerButton.setBounds(formLeftX + labelWidth + 20 + 180, 640, 160, 45);
        registerButton.addActionListener(e -> openRegisterWindow());
        bgLabel.add(registerButton);

        // 状态提示文字
        statusLabel = new JLabel("");
        statusLabel.setForeground(Color.YELLOW);
        statusLabel.setFont(statusFont);
        statusLabel.setBounds(formLeftX, 700, 460, 35);
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
            // TODO：这里可以打开主界面、关闭当前窗口
        } else {
            statusLabel.setText("用户名或密码错误！");
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

        Font labelFont = new Font("微软雅黑", Font.BOLD, 20);
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
        userLabel.setBounds(leftX, 60, labelWidth, height);
        add(userLabel);

        usernameField = new JTextField();
        usernameField.setFont(fieldFont);
        usernameField.setBounds(leftX + labelWidth + 10, 60, fieldWidth, height);
        add(usernameField);

        // 密码
        JLabel passLabel = new JLabel("密码：");
        passLabel.setFont(labelFont);
        passLabel.setBounds(leftX, 110, labelWidth, height);
        add(passLabel);

        passwordField = new JPasswordField();
        passwordField.setFont(fieldFont);
        passwordField.setBounds(leftX + labelWidth + 10, 110, fieldWidth, height);
        add(passwordField);

        // 确认密码
        JLabel confirmLabel = new JLabel("确认密码：");
        confirmLabel.setFont(labelFont);
        confirmLabel.setBounds(leftX, 160, labelWidth + 40, height);
        add(confirmLabel);

        confirmField = new JPasswordField();
        confirmField.setFont(fieldFont);
        confirmField.setBounds(leftX + labelWidth + 50, 160, fieldWidth, height);
        add(confirmField);

        // 提交按钮
        submitButton = new JButton("提交注册");
        submitButton.setFont(buttonFont);
        submitButton.setBounds(leftX + 80, 220, 200, 40);
        submitButton.addActionListener(this);
        add(submitButton);

        // 状态提示
        statusLabel = new JLabel("");
        statusLabel.setFont(statusFont);
        statusLabel.setForeground(Color.RED);
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
            // 常见：UNIQUE 约束冲突（用户名重复）
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

    // 示例：你以后想直接写 SQL 也可以扩展，例如：
    // public static void runCustomQuery(String sql) { ... }
}
