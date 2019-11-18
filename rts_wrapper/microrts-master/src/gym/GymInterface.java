package gym;

import ai.abstraction.WorkerRush;
import ai.abstraction.pathfinding.BFSPathFinding;
import ai.core.AI;
import ai.*;
import gui.PhysicalGameStatePanel;

import javax.swing.JFrame;

import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.UnitTypeTable;
import weka.core.pmml.jaxbbindings.True;

import java.io.*;

/**
 * @author santi, Tom
 */

public class GymInterface {
    private static long maxEpisodes = 100000;
    private static int timeBudget = 100;
    private static long maxCycles = 20000;
    private static int period = 1;
    private static long port = 9898;
    private static String map = System.getProperty("user.home") + "/microrts_env/maps/16x16/basesWorkers16x16.xml";
    private static int skipFrame = 10;
//    private static Writer outWriter = new BufferedWriter(new OutputStreamWriter(System.out));

    private static void parseArgs(String[] args){
        System.out.println("Client received info:");

        for (String arg : args) {
            System.out.println(arg);
        }

        CliArgs cliArgs = new CliArgs(args);
        port = cliArgs.switchPresent("--port") ? Long.parseLong(cliArgs.switchValue("--port")) : port;
        map = cliArgs.switchPresent("--map") ? cliArgs.switchValue("--map") : map;
        maxEpisodes = cliArgs.switchPresent("--maxEpisodes") ? Long.parseLong(cliArgs.switchValue("--maxEpisodes")) : maxEpisodes;
        maxCycles = cliArgs.switchPresent("--maxCycles") ? Long.parseLong(cliArgs.switchValue("--maxCycles")) : maxCycles;

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
        GymSocketAI ai1 = new GymSocketAI(timeBudget, 0, "127.0.0.1", (int) port, GymSocketAI.LANGUAGE_JSON, utt);
        AI ai2 = new RandomAI();


        for (int i = 0; i < maxEpisodes; i++) {
            PhysicalGameState pgs = PhysicalGameState.load(map, utt);
            GameState gs = new GameState(pgs, utt);

            boolean gameover;
            ai1.reset(gs, 0);
            ai2.reset();

            JFrame w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, PhysicalGameStatePanel.COLORSCHEME_BLACK);
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
                    ai1.sendGameState(gs,0,false,done);

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

            System.out.println("Done");
            ai1.send_winner(gs);
//            ai1.close();
            w.dispose();
        }

//        ai1.reset();

    }
}
