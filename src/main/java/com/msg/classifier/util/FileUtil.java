package com.msg.classifier.util;

import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by MSG on 2018-04-12:上午9:16
 */
@Slf4j
public class FileUtil {

    public static final String DEFAULT_ENCODE = "utf-8";
    public static final String DEFAULT_VALUE = "";
    public static final String DEFAULT_IGNORE = "#";

    private FileUtil() {

    }


    public static List<String> readFileToArray(String filePath) {
        return readFileToArray(filePath, DEFAULT_ENCODE);
    }

    public static List<String> readFileToArray(String filePath, String encode) {
        return readFileToArray(filePath, encode, DEFAULT_IGNORE);
    }

    /**
     * 将文件里的内容按行读取到一个list列表中
     *
     * @param filePath 文件路径
     * @param encode   文件编码
     * @param ignore   以这个字符开始的行，跳过
     * @return
     */
    public static List<String> readFileToArray(String filePath, String encode, String ignore) {
        List<String> lines = new ArrayList<>();
        encode = (null == encode || DEFAULT_VALUE.equals(encode)) ? DEFAULT_ENCODE : encode;
        BufferedReader bufferedReader = null;
        String line;
        try {
            bufferedReader = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(filePath), encode));
            while (null != (line = bufferedReader.readLine())) {
                if (!DEFAULT_VALUE.equals(line) && !line.startsWith(ignore)) {
                    lines.add(line);
                }
            }
        } catch (IOException e) {
            log.error("读取文件出错:" + filePath);
        } finally {
            if (null != bufferedReader) {
                try {
                    bufferedReader.close();
                } catch (IOException e) {
                    log.error("关闭缓冲流出错！");
                }
            }
        }
        return lines;
    }

    public static void writeStrToFile(String line, String filePath) {
        writeStrToFile(line, filePath, DEFAULT_ENCODE, false);
    }

    public static void writeStrToFile(String line, String filePath, String encode, boolean append) {
        List<String> lines = Arrays.asList(line);
        writeArrayToFile(lines, filePath, encode, append);
    }

    public static <T> void writeArrayToFile(List<T> lines, String filePath) {
        writeArrayToFile(lines, filePath, DEFAULT_ENCODE, false);
    }

    public static <T> void writeArrayToFile(List<T> lines, String filePath, String encode) {
        writeArrayToFile(lines, filePath, encode, false);
    }


    /**
     * 将列表数据写入文件
     *
     * @param lines    列表数据
     * @param filePath 写出的路径
     * @param encode   写出的编码
     * @param append   是否添加模式
     */
    public static <T> void writeArrayToFile(List<T> lines, String filePath, String encode, boolean append) {
        encode = (null == encode || DEFAULT_VALUE.equals(encode)) ? DEFAULT_ENCODE : encode;
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filePath, append), encode));
            for (Object line : lines) {
                bw.write(line.toString());
                bw.newLine();
            }
        } catch (FileNotFoundException e) {
            log.error("文件路径出错:" + filePath);
        } catch (UnsupportedEncodingException e) {
            log.error("不支持的编码格式:" + encode);
        } catch (IOException e) {
            log.error("写入出错:" + filePath);
        } finally {
            if (null != bw) {
                try {
                    bw.close();
                } catch (IOException e) {
                    log.error("关闭输出缓冲流出错");
                }
            }
        }
    }


    public static File createFile(String path) {
        return createPath(path, "FILE");
    }

    public static File createDir(String path) {
        return createPath(path, "DIR");
    }

    /**
     * 创建目录或者文件
     *
     * @param path
     * @param type
     * @return
     */
    private static File createPath(String path, String type) {
        File file = new File(path);
        if (!file.exists()) {
            if ("FILE".equals(type)) {
                try {
                    if (file.createNewFile()) {
                        log.info("创建文件成功:" + path);
                    }
                } catch (IOException e) {
                    log.error("文件出错:" + path);
                }
            } else if ("DIR".equals(type)) {
                if (file.mkdirs()) {
                    log.info("创建目录成功:" + path);
                }
            } else {
                log.error("输入类型出错，只能FILE或者DIR");
            }
        }
        return file;
    }

    /**
     * 简单的获得一个目录下所有的文件
     *
     * @param dir
     * @return
     */
    public static List<String> listFiles(String dir) {
        List<String> filesList = new ArrayList<>();
        File path = new File(dir);
        if (path.isDirectory()) {
            File[] files = path.listFiles();
            if(null != files) {
                for (File file : files) {
                    filesList.addAll(listFiles(file.getAbsolutePath()));
                }
            }
        } else {
            filesList.add(path.getAbsolutePath());
        }
        return filesList;
    }


}
