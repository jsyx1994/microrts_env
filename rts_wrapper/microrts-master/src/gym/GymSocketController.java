package gym;

import rts.units.UnitTypeTable;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.net.Socket;

public class GymSocketController {
    public static int DEBUG = 1;

    public static final int LANGUAGE_XML = 1;
    public static final int LANGUAGE_JSON = 2;

    UnitTypeTable utt = null;

    int communication_language = LANGUAGE_XML;
    String serverAddress = "127.0.0.1";
    int serverPort = 9898;
    Socket socket = null;
    BufferedReader in_pipe = null;
    PrintWriter out_pipe = null;
}
