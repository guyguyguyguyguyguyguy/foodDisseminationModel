import com.sun.nio.sctp.AbstractNotificationHandler;

import java.awt.*;
import java.io.*;
import java.awt.event.*; // Using AWT event classes and listener interfaces
import javax.sound.midi.SysexMessage;
import javax.swing.*;    // Using Swing components and containers
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import java.io.IOException;
import java.lang.reflect.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Flow;


// A Swing GUI application inherits the top-level container javax.swing.JFrame
public class abm_gui extends JFrame {
    public JTextField tfRepeats, tfSteps, tfAboveBias, tfBelowBias,
            tfFInertia, tfBInertia, tfDepth, tfHeight, tfNestMove, tfAboveVel,
            tfBelowVel, tfNestIntRate, tfForLag;
    public JComboBox<String> tfSetUp;
    public JTextArea saveText, pythonPath, abmPath, tfparrallelize, tffileName;
    public JButton saveBut, pythonButton, abmButton;
    public JButton tfDone;
    public JCheckBox tfInertia, tfSave, tfVerbose, tfNestBiasMov, tfShuffle, tfGiveEveryStep;
    public JList<String> tfDim, tfTroph, tfMov;
    String steps = "5000";
    String forward_inertia = "0";
    String backward_inertia = "0";
    String bias_above;
    String bias_below;
    String vel_above = "0.05";
    String vel_below = "-0.25";
    String repeats = "2";
    String verbose = "0";
    String giveAtEveryStep = "0";
    String save;
    String filename;
    String shuffleAtExit = "0";
    String nestAntBehaviour;
    String nestmate_movement = "0";
    String nest_depth = "45";
    String nest_height = "1";
    String troph;
    String move;
    String lag_len = "1";
    String nestmate_bias = "0";
    String nestmate_int_rate = "0";
    String python;
    String abm;
    String parrallelize = "1";



    // Constructor to setup the GUI components and event handlers
    public abm_gui() {

        // Retrieve the content-pane of the top-level container JFrame
        // All operations done on the content-pane
        Container cp = getContentPane();
        cp.setLayout(new GridLayout(15, 4, 5, 5));  // The content-pane sets
        // its layout
//        cp.setLayout(null);  // The content-pane sets its layout

        JPanel firstPanel = new JPanel();
        firstPanel.setLayout(new GridLayout(0, 5, 10, 10));
        cp.add(firstPanel);
        firstPanel.add(new JLabel("Number of steps: "));
        tfSteps = new JTextField(1);
        tfSteps.setText(String.valueOf(steps));
        firstPanel.add(tfSteps);
        firstPanel.add(new JLabel("Number of repeats: "));
        tfRepeats = new JTextField(1);
        tfRepeats.setText(String.valueOf(repeats));
//        tfOutput.setEditable(true);  // read-only
        firstPanel.add(tfRepeats);


        JPanel secondPanel = new JPanel();
        secondPanel.setLayout(new GridLayout(0, 5, 20, 10));
        cp.add(secondPanel);
        secondPanel.add(new JLabel("Nest depth"));
        tfDepth = new JTextField(1);
        tfDepth.setText(String.valueOf(nest_depth));
        secondPanel.add(tfDepth);
        secondPanel.add(new JLabel("Nest height"));
        tfHeight = new JTextField(1);
        tfHeight.setText(String.valueOf(nest_height));
        secondPanel.add(tfHeight);


//        JPanel thirdPanel = new JPanel();
//        thirdPanel.setLayout(new GridLayout(0, 2, 20, 10));
//        cp.add(thirdPanel);
//        String[] setUp = { "space", "order", "homogenise"};
//        tfSetUp = new JComboBox<>(setUp);
//        thirdPanel.add(tfSetUp);


        JPanel fourthPanel = new JPanel();
        fourthPanel.setLayout(new GridLayout(0, 5, 20, 10));
        cp.add(fourthPanel);
        fourthPanel.add(new JLabel("Trophallaxis type: "));
        String[] tropOp = { "Stochastic", "Deterministic" };
        tfTroph = new JList<>(tropOp);
        fourthPanel.add(tfTroph);
        fourthPanel.add(new JLabel("Movement type: "));
        String[] movOp = { "Stochastic", "Deterministic", "Spaceless", "Average-velocity", "TwoD-Stochastic" };
        tfMov = new JList<>(movOp);
        fourthPanel.add(tfMov);


        JPanel fifthPanel = new JPanel();
        fifthPanel.setLayout(new GridLayout(0, 6, 20, 10));
        cp.add(fifthPanel);
        fifthPanel.add(new JLabel("Forager biases"));
        tfAboveBias = new JTextField(1);
        tfAboveBias.setText("[0.3, 0.65, 1]");
        fifthPanel.add(new JLabel("Above thresh: "));
        fifthPanel.add(tfAboveBias);
        tfBelowBias = new JTextField(1);
        tfBelowBias.setText("[0.5, 0.75, 1]");
        fifthPanel.add(new JLabel("Below thresh: "));
        fifthPanel.add(tfBelowBias);


        JPanel sixthpanel = new JPanel();
        sixthpanel.setLayout(new GridLayout(0, 6, 20, 10));
        cp.add(sixthpanel);
        sixthpanel.add(new JLabel("Forager velocity"));
        tfAboveVel = new JTextField(2);
        tfAboveVel.setText(String.valueOf(vel_above));
        sixthpanel.add(new JLabel("Above thresh: "));
        sixthpanel.add(tfAboveVel);
        tfBelowVel = new JTextField(2);
        tfBelowVel.setText(String.valueOf(vel_below));
        sixthpanel.add(new JLabel("Below thresh: "));
        sixthpanel.add(tfBelowVel);



//        String[] dims = { "1D", "2D" };
//        tfDim = new JList<>(dims);
//        cp.add(tfDim);
//

        JPanel sevenpanel = new JPanel();
        sevenpanel.setLayout(new GridLayout(0, 5, 20, 10));
        cp.add(sevenpanel);
        sevenpanel.add(new JLabel("Forward inertia"));
        tfFInertia = new JTextField(4);
        tfFInertia.setText(String.valueOf(forward_inertia));
        sevenpanel.add(tfFInertia);
        sevenpanel.add(new JLabel("Backward inertia"));
        tfBInertia = new JTextField(4);
        tfBInertia.setText(String.valueOf(backward_inertia));
        sevenpanel.add(tfBInertia);


        JPanel checkPanel = new JPanel();
        checkPanel.setLayout(new FlowLayout());
        cp.add(checkPanel);
        tfVerbose = new JCheckBox("Verbose");
        checkPanel.add(tfVerbose);
        tfNestBiasMov = new JCheckBox("Emptier non-foragers move towards " +
                "entrance");
        checkPanel.add(tfNestBiasMov);
        tfShuffle = new JCheckBox("Shuffle nest at forager exit");
        checkPanel.add(tfShuffle);
        tfGiveEveryStep = new JCheckBox("Give at every step (deterministic movement)");
        checkPanel.add(tfGiveEveryStep);
       


        JPanel nonForPanel = new JPanel();
        nonForPanel.setLayout(new FlowLayout());
        cp.add(nonForPanel);
        nonForPanel.add(new JLabel("Interaction rate of non-foragers"));
        tfNestIntRate = new JTextField(4);
        tfNestIntRate.setText(String.valueOf(nestmate_int_rate));
        nonForPanel.add(tfNestIntRate);
        nonForPanel.add(new JLabel("Cells non-foragers can move at forager exit: "));
        tfNestMove = new JTextField(4);
        tfNestMove.setText(String.valueOf(nestmate_movement));
        nonForPanel.add(tfNestMove);
        nonForPanel.add(new JLabel("Lag of forager when at entrance"));
        tfForLag = new JTextField(4);
        tfForLag.setText(String.valueOf(lag_len));
        nonForPanel.add(tfForLag);


        JPanel parallel = new JPanel();
        parallel.setLayout(new FlowLayout());
        cp.add(parallel);
        parallel.add(new JLabel("Number of cores to run on: "));
        tfparrallelize = new JTextArea(2, 20);
        tfparrallelize.setText(parrallelize);
        parallel.add(tfparrallelize);

        JPanel pathPython = new JPanel();
        pathPython.setLayout(new FlowLayout());
        cp.add(pathPython);
        pathPython.add(new JLabel("Path to python"));
        pythonPath = new JTextArea(2, 40);
        pythonButton = new JButton("Find path");
        pathPython.add(new JScrollPane(pythonPath)) ;
        pathPython.add(pythonButton);


        JPanel pathABM = new JPanel();
        pathABM.setLayout(new FlowLayout());
        cp.add(pathABM);
        pathABM.add(new JLabel("Path to runSimulation"));
        abmPath = new JTextArea(2, 40);
        abmButton = new JButton("Find path");
        pathABM.add(new JScrollPane(abmPath)) ;
        pathABM.add(abmButton);


        JPanel saveNamePanel = new JPanel();
        saveNamePanel.setLayout(new FlowLayout());
        cp.add(saveNamePanel);
        saveNamePanel.add(new JLabel("Save run as : "));
        tffileName =  new JTextArea(2, 40);
        saveNamePanel.add(tffileName);

        JPanel savePanel = new JPanel();
        savePanel.setLayout(new FlowLayout());
        cp.add(savePanel);
        savePanel.add(new JLabel("Folder to save to: "));
        saveText = new JTextArea(2, 40);
        saveBut = new JButton("Save to file");
        savePanel.add(new JScrollPane(saveText)) ;
        savePanel.add(saveBut);

        JPanel donePanel = new JPanel();
        donePanel.setLayout(new FlowLayout());
        cp.add(donePanel);
        tfDone = new JButton("Run");
        donePanel.add(tfDone);


        saveBut.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                saveToFile(saveText);
            }
        });

        pythonButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                selectFile(pythonPath);
            }
        });

        abmButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                selectFile(abmPath);
            }
        });


        tfDone.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Can execute python program here with the parameters
                // Need to add validation to this
                steps = tfSteps.getText();
                repeats = tfRepeats.getText();
                bias_above = tfAboveBias.getText();
                bias_below = tfBelowBias.getText();
                forward_inertia = tfFInertia.getText();
                backward_inertia = tfBInertia.getText();
                vel_above = tfAboveVel.getText();
                vel_below = tfBelowVel.getText();
                verbose = String.valueOf(tfVerbose.isSelected() ? 1:0);
                shuffleAtExit = String.valueOf(tfShuffle.isSelected() ? 1:0);
                giveAtEveryStep = String.valueOf(tfGiveEveryStep.isSelected() ? 1:0);
                save = saveText.getText();
                filename = tffileName.getText();
                nestmate_movement = tfNestMove.getText();
                nest_depth = tfDepth.getText();
                nest_height = tfHeight.getText();
                troph = tfTroph.getSelectedValue();
                move = tfMov.getSelectedValue();
                lag_len = tfForLag.getText();
                nestmate_bias = String.valueOf(tfNestBiasMov.isSelected() ?
                        1:0);
                nestmate_int_rate = tfNestIntRate.getText();
                python = pythonPath.getText();
                abm = abmPath.getText();
                parrallelize = tfparrallelize.getText();


                abm_gui.super.dispose();

                // Run python/bash code here
                String[] cmd = new String[]{python, abm, "from_gui", steps,
                        repeats,
                        bias_above, bias_below, forward_inertia,
                        backward_inertia, vel_above, vel_below, verbose, giveAtEveryStep,
                        shuffleAtExit, save
                        , filename, nestmate_movement, nest_depth, nest_height,
                        troph,
                        move, lag_len, nestmate_bias, nestmate_int_rate,
                        parrallelize};

                String s = null;

                try {

                    // Todo: Need to retrieve parameters in the python code
//                    System.out.println(Arrays.toString(cmd));
                    Process p = Runtime.getRuntime().exec(cmd);

                    BufferedReader stdInput = new BufferedReader(new
                            InputStreamReader(p.getInputStream()));

                    System.out.println("Here is the standard output of the command:\n");
                    while ((s = stdInput.readLine()) != null) {
                        System.out.println(s);
                    }

                    } catch (IOException ioException) {
                        ioException.printStackTrace();
                    }

                    System.exit(0);
            }
        });



        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);  // Exit program if close-window button clicked
        setTitle("Trophallaxis abm parameters"); // "super" Frame sets title
        setSize(1250, 820);  // "super" Frame sets initial size
        setVisible(true);   // "super" Frame shows
    }

    protected void saveToFile(JTextArea x) {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        fileChooser.showSaveDialog(saveBut);
        File dirr = fileChooser.getSelectedFile();
        x.setText(String.valueOf(dirr));

    }

    protected void selectFile(JTextArea x) {
        JFileChooser fileChooser = new JFileChooser();
//        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        fileChooser.showSaveDialog(saveBut);
        File dirr = fileChooser.getSelectedFile();
        x.setText(String.valueOf(dirr));

    }

    protected ArrayList<Float> stringArray(String lst){
        String debracketed = lst.replace("[", "").replace("]", "");
        String trimmed = debracketed.replaceAll("\\s+", "");
        ArrayList<String> list = new ArrayList<String>(Arrays.asList(trimmed.split(",")));
        ArrayList<Float> intList = new ArrayList<>();
        for(String s : list) intList.add(Float.valueOf(s));

        return intList;
    }

}
