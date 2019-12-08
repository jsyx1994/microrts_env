package gym;

import ai.abstraction.WorkerRush;
import ai.abstraction.pathfinding.BFSPathFinding;
import ai.core.AI;
import ai.*;
import ai.mcts.naivemcts.NaiveMCTS;
import gui.PhysicalGameStatePanel;

import javax.swing.JFrame;

import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.UnitTypeTable;
import tests.MapGenerator;
import weka.core.pmml.jaxbbindings.False;
import weka.core.pmml.jaxbbindings.True;

import java.io.*;

/**
 * @author santi, Tom
 */

public class GymInterface {
    private static long maxEpisodes;
    private static int timeBudget = 100;
    private static long period = 1;

    private static long maxCycles;
    private static long port1;
    private static long port2;
    private static String map;
    private static int skipFrame;
    private static String ai1_type;
    private static String ai2_type;
    private static long render;


//    private static Writer outWriter = new BufferedWriter(new OutputStreamWriter(System.out));

    private static void parseArgs(String[] args) {
        System.out.println("Client received info:");

        for (String arg : args) {
            System.out.println(arg);
        }

        CliArgs cliArgs = new CliArgs(args);
        port1 = Long.parseLong(cliArgs.switchValue("--port1", "9898"));
        port2 = Long.parseLong(cliArgs.switchValue("--port2", "8787"));
        map = cliArgs.switchValue("--map", System.getProperty("user.home") + "/microrts_env/maps/16x16/basesWorkers16x16.xml");
        maxEpisodes = Long.parseLong(cliArgs.switchValue("--maxEpisodes", "20000"));
        maxCycles = Long.parseLong(cliArgs.switchValue("--maxCycles", "5000"));
        period = Long.parseLong(cliArgs.switchValue("--period", "1"));
        render = Long.parseLong(cliArgs.switchValue("--render", "1"));


        ai1_type = cliArgs.switchValue("--ai1_type","Passive");
        ai2_type = cliArgs.switchValue("--ai2_type","Passive");

//        port = cliArgs.switchPresent("--port") ? Long.parseLong(cliArgs.switchValue("--port")) : port;
//        map = cliArgs.switchPresent("--map") ? cliArgs.switchValue("--map") : map;
//        maxEpisodes = cliArgs.switchPresent("--maxEpisodes") ? Long.parseLong(cliArgs.switchValue("--maxEpisodes")) : maxEpisodes;
//        maxCycles = cliArgs.switchPresent("--maxCycles") ? Long.parseLong(cliArgs.switchValue("--maxCycles")) : maxCycles;
//        ai1_type = cliArgs.switchPresent("--ai1_type") ? cliArgs.switchValue("--ai1_type") : ai1_type;
//        ai2_type = cliArgs.switchPresent("--ai2_type") ? cliArgs.switchValue("--ai2_type") : ai2_type;

    }

    private static void socketVSbuiltIn(UnitTypeTable utt) throws Exception {
        GymSocketAI ai1 = new GymSocketAI(timeBudget, 0, "127.0.0.1", (int) port1, GymSocketAI.LANGUAGE_JSON, utt);
        AI ai2 = new PassiveAI();
        switch (ai2_type){
            case "WorkerRush"   : ai2 = new WorkerRush(utt);    break;
            case "Random"       : ai2 = new RandomAI();         break;
            case "RandomBiased" : ai2 = new RandomBiasedAI(utt);break;
            case "NaiveMCTS"    : ai2 = new NaiveMCTS(utt);     break;
//            default:            : ai2 = new PassiveAI();        break;
        }

        for (int i = 0; i < maxEpisodes; i++) {
            PhysicalGameState pgs = PhysicalGameState.load(map, utt);
            GameState gs = new GameState(pgs, utt);

            boolean gameover;
            ai1.reset(gs, 0);
            ai2.reset();
            JFrame w = null;
            if (render==1){
                w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, PhysicalGameStatePanel.COLORSCHEME_BLACK);
            }
//        JFrame w = PhysicalGameStatePanel.newVisualizer(gs,640,640,false,PhysicalGameStatePanel.COLORSCHEME_WHITE);

//        ai1.preGameAnalysis(gs, 1000, ".");
//        ai2.preGameAnalysis(gs, 1000, ".");

            boolean done = false;   //gym signal
            long nextTimeToUpdate = System.currentTimeMillis() + period;
            do {
                if (System.currentTimeMillis() >= nextTimeToUpdate) {
                    PlayerAction pa1 = ai1.getAction(0, gs);
                    PlayerAction pa2 = ai2.getAction(1, gs);
                    gs.issueSafe(pa1);
                    gs.issueSafe(pa2);

                    // simulate:
                    gameover = gs.cycle();
                    done = gameover || gs.getTime() >= maxCycles;
                    ai1.sendGameState(gs, 0, false, done);
                    if (render==1)
                        w.repaint();
                    nextTimeToUpdate += period;
                } else {
                    try {
                        Thread.sleep(1);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            } while (!done);

//            System.out.println("Done");
            ai1.send_winner(gs);
//            ai1.close();
            if (render==1)
                w.dispose();
        }
    }

    private static void socketVSSocket(UnitTypeTable utt) throws Exception {
        GymSocketAI ai1 = new GymSocketAI(timeBudget, 0, "127.0.0.1", (int) port1, GymSocketAI.LANGUAGE_JSON, utt);
        GymSocketAI ai2 = new GymSocketAI(timeBudget, 0, "127.0.0.1", (int) port2, GymSocketAI.LANGUAGE_JSON, utt);

//        AI ai2 = new PassiveAI();

        for (int i = 0; i < maxEpisodes; i++) {
            PhysicalGameState pgs = PhysicalGameState.load(map, utt);
            GameState gs = new GameState(pgs, utt);

            boolean gameover;
            ai1.reset(gs, 0);
            ai2.reset(gs, 1);
            JFrame w = null;
            if (render==1){
                w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, PhysicalGameStatePanel.COLORSCHEME_BLACK);
            }
//        JFrame w = PhysicalGameStatePanel.newVisualizer(gs,640,640,false,PhysicalGameStatePanel.COLORSCHEME_WHITE);

//        ai1.preGameAnalysis(gs, 1000, ".");
//        ai2.preGameAnalysis(gs, 1000, ".");

            boolean done = false;   //gym signal
            long nextTimeToUpdate = System.currentTimeMillis() + period;
            do {
                if (System.currentTimeMillis() >= nextTimeToUpdate) {
                    PlayerAction pa1 = ai1.getAction(0, gs);
                    PlayerAction pa2 = ai2.getAction(1, gs);
                    gs.issueSafe(pa1);
                    gs.issueSafe(pa2);

                    // simulate:
                    gameover = gs.cycle();
                    done = gameover || gs.getTime() >= maxCycles;
                    ai1.sendGameState(gs, 0, false, done);
                    ai2.sendGameState(gs, 1, false, done);
                    if (render==1)
                        w.repaint();
                    nextTimeToUpdate += period;
                } else {
                    try {
                        Thread.sleep(1);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            } while (!done);

//            System.out.println("Done");
            ai1.send_winner(gs);
            ai2.send_winner(gs);
//            ai1.close();
            if (render==1)
                w.dispose();
        }
    }

    private static void builtInVSBuiltIn(UnitTypeTable utt) throws Exception {
        render = 1;
        AI ai1 = new PassiveAI();
        AI ai2 = new PassiveAI();
        switch (ai1_type){
            case "WorkerRush"   : ai2 = new WorkerRush(utt);    break;
            case "Random"       : ai2 = new RandomAI();         break;
            case "RandomBiased" : ai2 = new RandomBiasedAI(utt);break;
            case "NaiveMCTS"    : ai2 = new NaiveMCTS(utt);     break;
//            default:            : ai2 = new PassiveAI();        break;
        }
        switch (ai2_type){
            case "WorkerRush"   : ai2 = new WorkerRush(utt);    break;
            case "Random"       : ai2 = new RandomAI();         break;
            case "RandomBiased" : ai2 = new RandomBiasedAI(utt);break;
            case "NaiveMCTS"    : ai2 = new NaiveMCTS(utt);     break;
//            default:            : ai2 = new PassiveAI();        break;
        }

        for (int i = 0; i < maxEpisodes; i++) {
            PhysicalGameState pgs = PhysicalGameState.load(map, utt);
            GameState gs = new GameState(pgs, utt);

            boolean gameover;
            ai1.reset();
            ai2.reset();
            JFrame w = null;
            if (render==1){
                w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, PhysicalGameStatePanel.COLORSCHEME_BLACK);
            }
//        JFrame w = PhysicalGameStatePanel.newVisualizer(gs,640,640,false,PhysicalGameStatePanel.COLORSCHEME_WHITE);

//        ai1.preGameAnalysis(gs, 1000, ".");
//        ai2.preGameAnalysis(gs, 1000, ".");

            boolean done = false;   //gym signal
            long nextTimeToUpdate = System.currentTimeMillis() + period;
            do {
                if (System.currentTimeMillis() >= nextTimeToUpdate) {
                    PlayerAction pa1 = ai1.getAction(0, gs);
                    PlayerAction pa2 = ai2.getAction(1, gs);
                    gs.issueSafe(pa1);
                    gs.issueSafe(pa2);

                    // simulate:
                    gameover = gs.cycle();
                    done = gameover || gs.getTime() >= maxCycles;
                    if (render==1)
                        w.repaint();
                    nextTimeToUpdate += period;
                } else {
                    try {
                        Thread.sleep(1);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            } while (!done);

//            ai1.close();
            if (render==1)
                w.dispose();
        }
    }

    public static void main(String[] args) throws Exception {
        parseArgs(args);
        //        AI ai1 = new WorkerRush(utt, new BFSPathFinding());
        //        AI ai1 = new GymSocketAI(100,0, "127.0.0.1", 9898, GymSocketAI.LANGUAGE_XML, utt);

        UnitTypeTable utt = new UnitTypeTable();

        StringWriter stringWriter = new StringWriter();
        utt.toJSON(stringWriter);
        System.err.println(stringWriter.toString());


//        PhysicalGameState pgs = MapGenerator.basesWorkers8x8Obstacle();

        //  AI ai2 = new WorkerRush(utt);

        if (ai1_type.equals("socketAI") && !ai2_type.equals("socketAI")) {
            socketVSbuiltIn(utt);
        } else if (ai1_type.equals("socketAI") && ai2_type.equals("socketAI")) {
            socketVSSocket(utt);
        } else if (!ai1_type.equals("socketAI") && !ai2_type.equals("socketAI")) {
            builtInVSBuiltIn(utt);
        }

//        ai1.reset();

    }
}
