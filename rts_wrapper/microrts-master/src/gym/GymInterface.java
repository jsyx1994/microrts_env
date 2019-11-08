package gym;

import ai.abstraction.WorkerRush;
import ai.abstraction.pathfinding.BFSPathFinding;
import ai.core.AI;
import ai.*;
import ai.socket.SocketAI;
import gui.PhysicalGameStatePanel;

import javax.swing.JFrame;

import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

import java.util.ArrayList;
import java.util.List;

/**
 * @author santi, Tom
 */

public class GymInterface {
    private static int MAXCYCLES = 5000;
    private static int PERIOD = 20;
    private static long port = 9898;
    private static String map = System.getProperty("user.home") + "/microrts_env/maps/16x16/basesWorkers16x16.xml";

    
    public static void main(String args[]) throws Exception {
//        for (int i = 0; i < args.length; ++i){
//            System.out.println(args[i]);
//        }
        CliArgs cliArgs = new CliArgs(args);
        port = cliArgs.switchPresent("--port") ? cliArgs.switchLongValue("--port") : port;
        map = cliArgs.switchPresent("--map") ? cliArgs.switchValue("--map") : map;

        System.out.println("Client received info:");
        System.out.println(port);
        System.out.println(map);


        UnitTypeTable utt = new UnitTypeTable();

        PhysicalGameState pgs = PhysicalGameState.load(map, utt);
//        PhysicalGameState pgs = MapGenerator.basesWorkers8x8Obstacle();

        GameState gs = new GameState(pgs, utt);
        boolean gameover = false;

//        AI ai1 = new WorkerRush(utt, new BFSPathFinding());
//        AI ai1 = new SocketAI(100,0, "127.0.0.1", 9898, SocketAI.LANGUAGE_XML, utt);
        AI ai1 = new GymSocketAI(100, 0, "127.0.0.1", (int)port, SocketAI.LANGUAGE_JSON, utt);
        AI ai2 = new RandomBiasedAI();

        ai1.reset();
        ai2.reset();

        JFrame w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, PhysicalGameStatePanel.COLORSCHEME_BLACK);
//        JFrame w = PhysicalGameStatePanel.newVisualizer(gs,640,640,false,PhysicalGameStatePanel.COLORSCHEME_WHITE);

//        ai1.preGameAnalysis(gs, 1000, ".");
//        ai2.preGameAnalysis(gs, 1000, ".");

        long nextTimeToUpdate = System.currentTimeMillis() + PERIOD;
        do {
            if (System.currentTimeMillis() >= nextTimeToUpdate) {
                PlayerAction pa1 = ai1.getAction(0, gs);
                PlayerAction pa2 = ai2.getAction(1, gs);
                gs.issueSafe(pa1);
                gs.issueSafe(pa2);

                // simulate:
                gameover = gs.cycle();
                w.repaint();
                nextTimeToUpdate += PERIOD;
            } else {
                try {
                    Thread.sleep(1);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        } while (!gameover && gs.getTime() < MAXCYCLES);

        System.out.println("Game Over");
    }
}
